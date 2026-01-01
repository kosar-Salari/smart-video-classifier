"""
core/eboo_api.py
Eboo OCR/ASR gateway client (robust for audio transcription)

Key improvements:
- Retry with exponential backoff for transient failures (500, network).
- Upload strategies for ambiguous 'filehandle':
  1) multipart/form-data
  2) JSON with base64 (filehandle)
  3) x-www-form-urlencoded with base64 (filehandle)
- Audio convert + polling (checkconvert)
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


DEFAULT_GATEWAY = "https://www.eboo.ir/api/ocr/getway"


class EbooAPIError(RuntimeError):
    pass


@dataclass
class EbooClient:
    token: str
    gateway: str = DEFAULT_GATEWAY
    timeout_sec: int = 90

    # Retry policy
    max_retries: int = 5
    backoff_base_sec: float = 1.2

    def _sleep_backoff(self, attempt: int) -> None:
        # attempt starts at 1
        time.sleep(self.backoff_base_sec * (2 ** (attempt - 1)))

    def _post_json(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any] | str]:
        r = requests.post(self.gateway, json=payload, timeout=self.timeout_sec)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text

    def _post_form(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any] | str]:
        r = requests.post(self.gateway, data=payload, timeout=self.timeout_sec)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text

    def _post_multipart(self, data: Dict[str, Any], files: Dict[str, Any]) -> Tuple[int, Dict[str, Any] | str]:
        r = requests.post(self.gateway, data=data, files=files, timeout=self.timeout_sec)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text

    @staticmethod
    def _is_transient_http(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    @staticmethod
    def _status_str(resp: Any) -> str:
        if isinstance(resp, dict):
            return str(resp.get("Status", "")).strip()
        return ""

    @staticmethod
    def _raise_if_known_error(resp: Any) -> None:
        # طبق مستند: اینها وضعیت‌های خطا هستند
        if not isinstance(resp, dict):
            return
        status = str(resp.get("Status", "")).strip()

        error_like = {
            "FileCorrupted",
            "NoEnoughCredit",
            "UploadFailed",
            "UnkwonURL",
            "UnknowMethod",
            "TokenNotFound",
            "MethodNotAllowed",
            "FileDeletedFromServer",
            "ConvertError",
            "Data Not Found Yet",
            "IPError",
            "404",
            "500",
        }
        if status in error_like:
            raise EbooAPIError(f"Eboo error: {status} | payload={resp}")

    def checkcredit(self) -> Dict[str, Any]:
        payload = {"token": self.token, "command": "checkcredit"}
        last_err = None

        for attempt in range(1, self.max_retries + 1):
            try:
                code, resp = self._post_json(payload)
                if isinstance(resp, dict):
                    self._raise_if_known_error(resp)
                    return resp
                # اگر JSON نشد، transient حساب می‌کنیم
                last_err = f"Non-JSON response: HTTP {code} | {str(resp)[:300]}"
                if self._is_transient_http(code):
                    self._sleep_backoff(attempt)
                    continue
                raise EbooAPIError(last_err)
            except requests.RequestException as e:
                last_err = str(e)
                self._sleep_backoff(attempt)

        raise EbooAPIError(f"checkcredit failed after retries. Last error: {last_err}")

    def addfile_by_link(self, filelink: str) -> Dict[str, Any]:
        payload = {"token": self.token, "command": "addfile", "filelink": filelink}
        last_err = None

        for attempt in range(1, self.max_retries + 1):
            try:
                code, resp = self._post_json(payload)
                if isinstance(resp, dict):
                    # اگر سرور خودش Status=500 برگرداند
                    status = self._status_str(resp)
                    if status == "500":
                        last_err = f"Server returned Status=500 payload={resp}"
                        self._sleep_backoff(attempt)
                        continue
                    self._raise_if_known_error(resp)
                    return resp

                last_err = f"Non-JSON response: HTTP {code} | {str(resp)[:300]}"
                if self._is_transient_http(code):
                    self._sleep_backoff(attempt)
                    continue
                raise EbooAPIError(last_err)
            except requests.RequestException as e:
                last_err = str(e)
                self._sleep_backoff(attempt)

        raise EbooAPIError(f"addfile_by_link failed after retries. Last error: {last_err}")

    def addfile_by_upload(self, filepath: str) -> Dict[str, Any]:
        """
        Robust addfile for ambiguous 'filehandle' specification.

        Tries:
        1) multipart/form-data  (filehandle as file)
        2) JSON base64          (filehandle as base64 string)
        3) form base64          (filehandle as base64 string)

        Retries transient failures.
        """
        last_err: Optional[str] = None

        # Prepare file bytes once (for base64 fallbacks)
        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read()
        except OSError as e:
            raise EbooAPIError(f"Cannot open file: {filepath} ({e})") from e

        b64 = base64.b64encode(file_bytes).decode("ascii")
        filename = filepath.split("/")[-1].split("\\")[-1]

        for attempt in range(1, self.max_retries + 1):
            # Strategy 1: multipart
            try:
                with open(filepath, "rb") as f:
                    data = {"token": self.token, "command": "addfile"}
                    files = {"filehandle": (filename, f)}
                    code, resp = self._post_multipart(data=data, files=files)

                if isinstance(resp, dict):
                    status = self._status_str(resp)
                    if status == "500":
                        last_err = f"Status=500 payload={resp}"
                        self._sleep_backoff(attempt)
                    else:
                        self._raise_if_known_error(resp)
                        return resp
                else:
                    last_err = f"Non-JSON response (multipart): HTTP {code} | {str(resp)[:300]}"
                    if self._is_transient_http(code):
                        self._sleep_backoff(attempt)
                    else:
                        # non-transient: still try other strategies this attempt
                        pass
            except requests.RequestException as e:
                last_err = f"multipart request failed: {e}"
                self._sleep_backoff(attempt)

            # Strategy 2: JSON base64
            try:
                payload = {"token": self.token, "command": "addfile", "filehandle": b64}
                code, resp = self._post_json(payload)

                if isinstance(resp, dict):
                    status = self._status_str(resp)
                    if status == "500":
                        last_err = f"Status=500 payload={resp}"
                        self._sleep_backoff(attempt)
                    else:
                        self._raise_if_known_error(resp)
                        return resp
                else:
                    last_err = f"Non-JSON response (json-b64): HTTP {code} | {str(resp)[:300]}"
                    if self._is_transient_http(code):
                        self._sleep_backoff(attempt)
            except requests.RequestException as e:
                last_err = f"json-b64 request failed: {e}"
                self._sleep_backoff(attempt)

            # Strategy 3: form base64
            try:
                payload = {"token": self.token, "command": "addfile", "filehandle": b64}
                code, resp = self._post_form(payload)

                if isinstance(resp, dict):
                    status = self._status_str(resp)
                    if status == "500":
                        last_err = f"Status=500 payload={resp}"
                        self._sleep_backoff(attempt)
                    else:
                        self._raise_if_known_error(resp)
                        return resp
                else:
                    last_err = f"Non-JSON response (form-b64): HTTP {code} | {str(resp)[:300]}"
                    if self._is_transient_http(code):
                        self._sleep_backoff(attempt)
            except requests.RequestException as e:
                last_err = f"form-b64 request failed: {e}"
                self._sleep_backoff(attempt)

        raise EbooAPIError(f"addfile_by_upload failed after retries. Last error: {last_err}")

    def convert_audio(self, filetoken: str, language: str = "fa", resetdata: bool = False) -> Dict[str, Any]:
        payload = {"token": self.token, "command": "convert", "filetoken": filetoken, "language": language}
        if resetdata:
            payload["resetdata"] = "1"

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                code, resp = self._post_json(payload)
                if isinstance(resp, dict):
                    status = self._status_str(resp)
                    if status == "500":
                        last_err = f"Status=500 payload={resp}"
                        self._sleep_backoff(attempt)
                        continue
                    self._raise_if_known_error(resp)
                    return resp

                last_err = f"Non-JSON response: HTTP {code} | {str(resp)[:300]}"
                if self._is_transient_http(code):
                    self._sleep_backoff(attempt)
                    continue
                raise EbooAPIError(last_err)
            except requests.RequestException as e:
                last_err = str(e)
                self._sleep_backoff(attempt)

        raise EbooAPIError(f"convert_audio failed after retries. Last error: {last_err}")

    def checkconvert(self, filetoken: str) -> Dict[str, Any]:
        payload = {"token": self.token, "command": "checkconvert", "filetoken": filetoken}
        code, resp = self._post_json(payload)
        if isinstance(resp, dict):
            return resp
        raise EbooAPIError(f"checkconvert non-JSON response. HTTP {code}: {str(resp)[:300]}")

    def wait_for_audio_text(
        self,
        filetoken: str,
        poll_interval_sec: float = 3.0,
        max_wait_sec: int = 30 * 60,
    ) -> Dict[str, Any]:
        start = time.time()
        last_status = None
        last_payload = None

        while True:
            data = self.checkconvert(filetoken)
            last_payload = data
            last_status = str(data.get("Status", "")).strip()

            out = data.get("Output")
            if isinstance(out, str) and out.strip():
                return data

            # بعضی مواقع Data Not Found Yet طبیعی است
            if time.time() - start > max_wait_sec:
                raise EbooAPIError(
                    f"Timeout waiting for transcription. LastStatus={last_status} | payload={last_payload}"
                )

            time.sleep(poll_interval_sec)

    def deletefile(self, filetoken: str) -> Dict[str, Any]:
        payload = {"token": self.token, "command": "deletefile", "filetoken": filetoken}
        code, resp = self._post_json(payload)
        if isinstance(resp, dict):
            # deletefile اگر خطا هم بدهد، پروژه شما نباید بایستد
            return resp
        return {"Status": "NonJSON", "HTTP": code, "Body": str(resp)[:300]}

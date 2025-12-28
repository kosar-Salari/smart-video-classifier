import gradio as gr
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import tempfile
from pathlib import Path
import anthropic

# تنظیمات
ANTHROPIC_API_KEY = "your-api-key-here"  # کلید API رو اینجا بذار

def extract_audio(video_path):
    """استخراج صدا از ویدیو"""
    try:
        video = VideoFileClip(video_path)
        audio_path = tempfile.mktemp(suffix=".wav")
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        video.close()
        return audio_path, None
    except Exception as e:
        return None, f"خطا در استخراج صدا: {str(e)}"

def transcribe_audio(audio_path, language="fa-IR"):
    """تبدیل صدا به متن"""
    recognizer = sr.Recognizer()
    
    try:
        # تبدیل به فرمت مناسب
        audio = AudioSegment.from_wav(audio_path)
        
        # تقسیم به قطعات کوچکتر برای دقت بیشتر
        chunks = split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=audio.dBFS-14,
            keep_silence=500,
        )
        
        full_text = []
        
        for i, chunk in enumerate(chunks[:10]):  # فقط 10 قطعه اول برای سرعت
            chunk_path = tempfile.mktemp(suffix=".wav")
            chunk.export(chunk_path, format="wav")
            
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                try:
                    # تلاش برای فارسی
                    if language == "fa-IR":
                        text = recognizer.recognize_google(audio_data, language="fa-IR")
                    else:
                        text = recognizer.recognize_google(audio_data, language="en-US")
                    full_text.append(text)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    return None, f"خطا در سرویس گوگل: {str(e)}"
            
            os.remove(chunk_path)
        
        if not full_text:
            return None, "متنی شناسایی نشد"
        
        return " ".join(full_text), None
        
    except Exception as e:
        return None, f"خطا در تبدیل صدا به متن: {str(e)}"

def categorize_content(text, api_key):
    """دسته‌بندی محتوا با Claude AI"""
    if not api_key or api_key == "your-api-key-here":
        return {
            "category": "⚠️ API Key وارد نشده",
            "confidence": "0%",
            "description": "لطفاً کلید API خود را در کد وارد کنید",
            "keywords": []
        }
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""متن زیر را تحلیل کن و دسته‌بندی کن:

"{text}"

لطفاً این متن را در یکی از دسته‌های زیر قرار بده:
- آشپزی و غذا
- نظامی و دفاعی
- ورزشی
- آموزشی
- سرگرمی
- اخبار
- تکنولوژی
- سلامت و پزشکی
- سفر و گردشگری
- هنر و موسیقی
- سایر

پاسخ رو به این فرمت JSON بده:
{{
    "category": "نام دسته",
    "confidence": "درصد اطمینان",
    "description": "توضیح کوتاه درباره محتوا",
    "keywords": ["کلمه کلیدی 1", "کلمه کلیدی 2", "کلمه کلیدی 3"]
}}"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        response_text = message.content[0].text
        
        # پاک کردن markdown اگر وجود داشت
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        return result
        
    except Exception as e:
        return {
            "category": "خطا",
            "confidence": "0%",
            "description": f"خطا: {str(e)}",
            "keywords": []
        }

def process_video(video_file, language, api_key):
    """پردازش کامل ویدیو"""
    if video_file is None:
        return "❌ لطفاً یک ویدیو آپلود کنید", "", "", "", ""
    
    # مرحله 1: استخراج صدا
    status = "🎬 در حال استخراج صدا..."
    yield status, "", "", "", ""
    
    audio_path, error = extract_audio(video_file)
    if error:
        yield f"❌ {error}", "", "", "", ""
        return
    
    # مرحله 2: تبدیل به متن
    status = "🎤 در حال تبدیل صدا به متن..."
    yield status, "", "", "", ""
    
    lang_code = "fa-IR" if language == "فارسی" else "en-US"
    text, error = transcribe_audio(audio_path, lang_code)
    
    # پاک کردن فایل صوتی موقت
    try:
        os.remove(audio_path)
    except:
        pass
    
    if error:
        yield f"❌ {error}", "", "", "", ""
        return
    
    # مرحله 3: دسته‌بندی
    status = "🤖 در حال تحلیل محتوا با هوش مصنوعی..."
    yield status, text, "", "", ""
    
    result = categorize_content(text, api_key)
    
    # نتیجه نهایی
    category_emoji = {
        "آشپزی و غذا": "🍳",
        "نظامی و دفاعی": "⚔️",
        "ورزشی": "⚽",
        "آموزشی": "📚",
        "سرگرمی": "🎭",
        "اخبار": "📰",
        "تکنولوژی": "💻",
        "سلامت و پزشکی": "⚕️",
        "سفر و گردشگری": "✈️",
        "هنر و موسیقی": "🎨",
        "سایر": "📁"
    }
    
    emoji = category_emoji.get(result["category"], "📁")
    
    final_status = f"✅ تحلیل کامل شد!"
    category_result = f"{emoji} {result['category']}"
    confidence_result = result['confidence']
    description_result = result['description']
    keywords_result = ", ".join(result['keywords'])
    
    yield final_status, text, category_result, confidence_result, description_result

# طراحی رابط کاربری
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {
        font-family: 'Vazirmatn', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
""") as demo:
    
    gr.HTML("""
        <div class="header">
            <h1>🎥 سیستم تحلیل هوشمند ویدیو</h1>
            <p>آپلود کنید، تحلیل کنید، نتیجه بگیرید!</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ تنظیمات")
            video_input = gr.Video(label="📹 آپلود ویدیو")
            language_input = gr.Radio(
                choices=["فارسی", "انگلیسی"],
                value="فارسی",
                label="🌐 زبان ویدیو"
            )
            api_key_input = gr.Textbox(
                label="🔑 کلید API (Anthropic)",
                placeholder="sk-ant-...",
                type="password",
                value=ANTHROPIC_API_KEY
            )
            process_btn = gr.Button("🚀 شروع تحلیل", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 نتایج تحلیل")
            status_output = gr.Textbox(label="📍 وضعیت", interactive=False)
            
            with gr.Accordion("📝 متن استخراج شده", open=True):
                text_output = gr.Textbox(label="", lines=5, interactive=False)
            
            with gr.Row():
                category_output = gr.Textbox(label="📂 دسته‌بندی", interactive=False)
                confidence_output = gr.Textbox(label="🎯 اطمینان", interactive=False)
            
            description_output = gr.Textbox(label="💬 توضیحات", lines=3, interactive=False)
    
    gr.Markdown("""
    ---
    ### 📖 راهنمای استفاده:
    1. کلید API خود را از [console.anthropic.com](https://console.anthropic.com) دریافت کنید
    2. ویدیو خود را آپلود کنید
    3. زبان ویدیو را انتخاب کنید
    4. روی دکمه "شروع تحلیل" کلیک کنید
    
    ### 📦 وابستگی‌های مورد نیاز:
    ```bash
    pip install gradio moviepy SpeechRecognition pydub anthropic
    ```
    """)
    
    process_btn.click(
        fn=process_video,
        inputs=[video_input, language_input, api_key_input],
        outputs=[status_output, text_output, category_output, confidence_output, description_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
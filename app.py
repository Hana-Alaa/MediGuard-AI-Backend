import os
from chatbot_module import process_message, listen

# Voice conversation
def chatbot_voice():
    while True:
        user_message = listen()
        if not user_message:
            print("❌ ما فهمتش الصوت، حاول تاني...")
            continue

        if user_message.lower().strip() in ["quit", "خروج", "انهاء", "إنهاء"]:
            print("👋 باي باي")
            break

        print(f"🧑‍🦱 أنت: {user_message}")
        print(f"🤖 البوت: {process_message(user_message)}")

# Text conversation
def chatbot_text():
    while True:
        user_message = input("📝 اكتب سؤالك: ").strip()
        if user_message.lower() in ["quit", "خروج", "انهاء", "إنهاء"]:
            print("👋 باي باي")
            break

        if not user_message:
            continue

        print(f"🤖 البوت: {process_message(user_message)}")

# Main entry point
if __name__ == "__main__":
    print("💻 مرحباً بك في المساعد الصحي الذكي!")
    print("🔹 اسأل أي سؤال طبي أو اطلب تقريرك الصحي")

    mode = input("تختار (1) صوت أو (2) كتابة؟ ").strip()
    if mode == "1":
        chatbot_voice()
    else:
        chatbot_text()
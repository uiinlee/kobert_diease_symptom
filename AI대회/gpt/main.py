import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import os, pygame, threading, openai
import speech_recognition as sr
from gtts import gTTS
from mutagen.mp3 import MP3 # pip install mutagen
import datetime

# OpenAI API키는 제외함

class ElderlyCareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("사회문제해결 해커톤")
        self.geometry("1024x600")
        self.resizable(False,False)

        self.nanum_font = tkfont.Font(family="NanumBarunGothic", size=12)

        self.frames = {}
        self.create_frames()
        self.show_frame("MainPage")

    def create_frames(self):
        for F in (MainPage, TalkingPage):
            frame = F(self)
            self.frames[F.__name__] = frame
            frame.place(x=0, y=0, width = 1024, height = 600)

    def show_frame(self, page_name, image_path=None):
        frame = self.frames[page_name]

        frame.tkraise()
        if page_name == "TalkingPage":
            frame.start_voice_interaction()

    def open_talking(self):
        self.show_frame("TalkingPage")

# 배경화면 설정
def set_background(root, path):
    img= Image.open(path)
    bg_image = ImageTk.PhotoImage(img)
    background_label = tk.Label(root, image=bg_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    background_label.image = bg_image # 참조 유지


# 메인 화면
class MainPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.create_widgets()
    
    def create_widgets(self):
        set_background(self, "./image/background.png")
        self. create_main_top_bar()

        buttons_info = [
            ("./image/doctor.png", "상담하기", self.master.open_talking),
        ]

        button_size = 200
        # gap = (1024 - 3 * button_size) / 4
        text_height = 30  # 텍스트 높이

        screen_width = 1024
        screen_height = 600

        x = (screen_width - button_size) / 2
        y = (screen_height - button_size - text_height) / 2

        image_path, text, command = buttons_info[0]
        self.create_image_button_with_text(image_path, text, command, x, y, button_size, text_height)
    
    def create_main_top_bar(self):
        # 상단바 생성
        top_bar = tk.Frame(self, bg="#F2F2F2", height = 50)
        top_bar.pack(side = "top", fill="x")

        # 전원 끄기 버튼
        power_img = Image.open("C:/Users/aaron/Desktop/Capstone/Code/Main_program/image/power.png")
        power_photo = ImageTk.PhotoImage(power_img.resize((40,40), Image.LANCZOS))
        power_button = tk.Button(top_bar, image=power_photo, bg="#F2F2F2", command=self.ask_to_exit)
        power_button.image = power_photo
        power_button.pack(side = "right", padx=(20,30), pady=10)

        # 시간 표시 레이블
        self.time_label = tk.Label(top_bar, bg="#F2F2F2", font=self.master.nanum_font)
        self.time_label.pack(side = "left", padx= 20, pady = 10)
        update_time(self.time_label)

    def ask_to_exit(self):
        if messagebox.askyesno("프로그램 종료", "프로그램을 종료하시겠습니까?"):
            self.master.destroy()

    def create_image_button_with_text(self, image_path, text, command, x, y, button_size, text_height):
        shadow_offset = 5

        # 그림자 레이블 생성
        shadow = tk.Label(self, bg='black')
        shadow.place(x=x + shadow_offset, y=y + shadow_offset, width=button_size, height=button_size)

        # 이미지 버튼 생성
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        button = tk.Button(self, image=img, borderwidth=2, relief="solid", command=command)
        button.place(x=x, y=y, width=button_size, height=button_size)
        button.image = img  # 참조 유지

        # 텍스트 레이블 생성
        label_y = y + button_size + 10  # 텍스트의 y좌표
        label = tk.Label(self, text=text, bg='white', font=self.master.nanum_font)
        label.place(x=x, y=label_y, width=button_size, height=text_height)

        return button


# 대화하기 화면
class TalkingPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.voice_assistant = VoiceAssistant(self)
        self.voice_thread = None
        self.voice_active = False
        self.create_widgets()
        self.current_y = 0

    def create_widgets(self):
        set_background(self, "C:/Users/aaron/Desktop/Capstone/Code/Main_program/image/background.png")
        self.create_top_bar()

        self.chat_frame = tk.Frame(self, bg="#ADD8E6") # 하늘색 프레임
        self.chat_frame.place(relx = 0.08, rely=0.175, relwidth=0.6, relheight=0.75)
        self.chat_canvas = tk.Canvas(self.chat_frame, bg="#ADD8E6")
        self.chat_scrollbar = tk.Scrollbar(self.chat_frame, orient="vertical", command=self.chat_canvas.yview)
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        self.chat_scrollbar.pack(side="right", fill="y")
        self.chat_canvas.pack(side="left", fill="both", expand=True)

        self.chat_content_frame = tk.Frame(self.chat_canvas, bg="#ADD8E6")
        self.chat_canvas.create_window((0,0), window=self.chat_content_frame, anchor="nw")

        # 대화를 시작하기 위한 안내 메시지
        custom_font = tkfont.Font(family="NanumBarunGothic", size=15)

        self.start_chat_label = tk.Label(self, text = "대화를 시작하려면 '선생님'이라고 불러보세요.", font=custom_font)
        self.start_chat_label.pack()

        doctor_img = Image.open("./image/ai_doctor.png")
        doctor_photo = ImageTk.PhotoImage(doctor_img.resize((300,300), Image.LANCZOS))
        # photo = ImageTk.PhotoImage(img)

        image_label = tk.Label(self, image=doctor_photo, borderwidth=0, highlightthickness=0, bg="#FAFAD2")
        image_label.image = doctor_photo
        image_label.place(relx=0.84, rely=0.5, anchor="center")

    def update_scroll_region(self):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def start_speak_text(self, text):
        tts = gTTS(text=text, lang ='ko')
        filename = "output.mp3"
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()
        pygame.mixer.quit()

        os.remove(filename)

    def start_voice_interaction(self):
        if not self.voice_thread or not self.voice_thread.is_alive():
            self.voice_active = True
            self.voice_thread = threading.Thread(target=self.run_voice_interaction, daemon = True)
            self.voice_thread.start()

    def run_voice_interaction(self):
        # 음성 인식의 시작
        while self.voice_active:
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                audio = recognizer.listen(source)
                try:
                    transcription = recognizer.recognize_google(audio, language='ko-KR')
                    if transcription.lower() == "선생님":
                        self.display_user_message("선생님")
                        self.display_bot_message("네, 무엇을 도와드릴까요?")
                        response = "네, 무엇을 도와드릴까요?"
                        print(f"[의사] : {response}")
                        self.start_speak_text(response)
                        self.start_chat_label.pack_forget() # 안내 메시지 숨기기
                        self.voice_assistant.interact_with_user() # 대화 시작
                        break
                except Exception as e:
                    continue # 계속 듣기

    def reset_interaction(self):
        # 대화 상태 초기화
        self.voice_thread = None
        self.start_chat_label.pack()

    def stop_voice_interaction(self):
        self.voice_active = False 
        self.voice_thread = None

    def display_user_message(self, text):
        user_frame = tk.Frame(self.chat_canvas, bg="#FFFF00")
        user_label = tk.Label(user_frame, text=text, bg="#FFFF00", font=self.master.nanum_font, wraplength=250)
        user_label.pack(side="right", fill="both", expand=True, padx=(0,20))
        
        # 프레임의 크기를 업데이트하고 위치 계산
        user_frame.update_idletasks()
        frame_height = user_frame.winfo_reqheight()
        self.chat_canvas.create_window((self.chat_frame.winfo_width(), self.current_y), window=user_frame, anchor="ne")

        # 다음 위젯의 y 좌표를 업데이트
        self.current_y += frame_height

        self.update_scroll_region()

    def display_bot_message(self, text):
        bot_frame = tk.Frame(self.chat_canvas, bg="#FFFFFF")
        bot_label = tk.Label(bot_frame, text=text, bg="#FFFFFF", font=self.master.nanum_font, wraplength=250)
        bot_label.pack(side="left", fill="both", expand=True, padx=(20,0))
        
        # 프레임의 크기를 업데이트하고 위치 계산
        bot_frame.update_idletasks()
        frame_height = bot_frame.winfo_reqheight()
        self.chat_canvas.create_window((0, self.current_y), window=bot_frame, anchor="nw")

        # 다음 위젯의 y 좌표를 업데이트
        self.current_y += frame_height

        self.update_scroll_region()


    def reset_chat(self):
        for widget in self.chat_content_frame.winfo_children():
            widget.destroy()

    def create_top_bar(self):
        # 상단바 생성
        top_bar = tk.Frame(self, bg="#F2F2F2", height = 50)
        top_bar.pack(side = "top", fill="x")

        # 돌아가기 버튼
        back_img = Image.open("C:/Users/aaron/Desktop/Capstone/Code/Main_program/image/left_arrow.png")
        back_photo = ImageTk.PhotoImage(back_img.resize((40,40), Image.LANCZOS))
        back_button = tk.Button(top_bar, image=back_photo, bg="#F2F2F2", command = lambda: [self.stop_voice_interaction(), self.reset_interaction_and_return()])
        back_button.image = back_photo
        back_button.pack(side = "left", padx=(20,10), pady=10)

        # 돌아가기 텍스트
        back_label = tk.Label(top_bar, text="돌아가기", bg="#F2F2F2", font=self.master.nanum_font)
        back_label.pack(side="left", pady= 10)

        # 시간 표시 레이블
        self.time_label = tk.Label(top_bar, bg="#F2F2F2", font=self.master.nanum_font)
        self.time_label.pack(side = "right", padx= 20, pady = 10)
        update_time(self.time_label)

    # 돌아가기 버튼을 누르면 MainPage로 돌아가고 대화내역 초기화
    def reset_interaction_and_return(self):
        self.reset_chat()
        self.reset_interaction()
        self.master.show_frame("MainPage")
        


class VoiceAssistant:
    def __init__(self,talking_page):
        self.talking_page = talking_page
        openai.api_key = "API_KEY"
    
    def interact_with_user(self):
        # 대화 진행
        messages = [
            {"role": "system", "content": "You are a comprehensive health counseling chatbot encompassing the roles of a doctor, mental health counselor, nutritionist, exercise specialist, and drug counselor. As a doctor, you ask patients questions and identify health problems based on symptoms, and suggest treatment options including medication prescriptions, surgery, lifestyle adjustments, and physical therapy. The role of a doctor includes proceeding carefully with invasive tests or treatments based on explicit patient consent. It's also crucial to maintain effective communication with patients to clearly explain diagnoses, treatment plans, expected outcomes, and possible side effects. Protecting patients' personal information and medical records and maintaining confidentiality is important. For accurate diagnosis, thorough medical history taking and physical examination are necessary, and if needed, appropriate diagnostic tests should be conducted. You must consider the patient's condition and individual needs to establish the best treatment plan and offer multiple treatment options when possible. Acting with respect, honesty, and fairness towards patients, adhering to medical ethics, and participating in continuous education and training to acquire the latest medical knowledge and skills are essential. Collaborating with a multidisciplinary team to provide comprehensive care to patients, prioritizing patient safety to prevent medical accidents, and having a deep knowledge of medications to recommend the most appropriate and safe drug treatments are very important. This involves understanding and applying all information about drugs, including their mechanism of action, dosage, administration method, side effects, and interactions, considering the patient's diagnosis, health condition, allergies, and other medications. Additionally, doctors must continually learn about the latest medical research and drug development trends to offer the latest treatment options to patients. This requires continuously updating medical knowledge and exchanging information through collaboration with experts, academic conferences, and medical journals. Providing patient-centered care while prioritizing the safety and effectiveness of drug use is one of the doctor's key responsibilities. As a mental health counselor, you assess mental states and provide basic advice on stress, depression, and anxiety management. As a nutritionist, you assess eating habits and provide nutritional guidelines for health improvement and disease management. As an exercise specialist, you recommend exercise plans tailored to an individual's physical condition and health goals. As a drug counselor, you explain the purpose, usage, and side effects of drugs to help patients understand and manage their prescribed medications properly. You listen to patients' medical histories, request diagnostic tests if necessary for accurate diagnosis, and counsel on diseases, treatment options, and health management. Avoid suggesting specific hospital visits or providing detailed medical advice that you cannot know. Provide clear, concise, and positive guidance, and keep the conversation simple for elderly users to understand."}
        ]

        # 당신은 의사, 정신 건강 상담사, 영양사, 운동 전문가, 약물 상담사의 역할을 포괄하는 종합 건강 상담 챗봇입니다.
        # 의사로서, 환자에게 질문하고 증상을 바탕으로 건강 문제를 식별하며 약물 처방, 수술, 생활 습관 조정, 물리 치료를 포함한 치료 방안을 제안합니다. 의사로서의 역할은 환자의 명시적 동의를 바탕으로 침습적 검사나 치료를 신중하게 진행하는 것을 포함합니다. 이와 함께, 환자와의 효과적인 의사소통을 유지하여 진단, 치료 계획, 예상되는 결과 및 가능한 부작용을 명확히 설명해야 합니다. 환자의 개인 정보와 의료 기록을 안전하게 보호하고 비밀을 유지하는 것도 중요합니다. 정확한 진단을 위해서는 철저한 병력 청취와 신체 검사, 필요 시 적절한 진단 검사를 시행해야 합니다. 환자의 상태와 개별적인 요구를 고려하여 최적의 치료 계획을 수립하며, 가능한 경우 여러 치료 옵션을 제공해야 합니다. 의료 윤리를 준수하며 환자에 대한 존중, 정직, 공정성을 바탕으로 행동해야 하며, 최신 의료 지식과 기술을 습득하기 위해 지속적인 교육과 훈련에 참여해야 합니다. 다학제 팀과의 협력을 통해 환자에게 포괄적인 관리를 제공하고, 환자의 안전을 최우선으로 고려하여 의료 관련 사고를 예방해야 합니다. 추가적으로, 의사로서 약물에 대한 깊은 지식을 갖추고, 환자의 상태와 필요에 맞는 약물을 추천하는 것이 매우 중요합니다. 이는 약물의 작용 기전, 용량, 투여 방법, 부작용, 상호작용 등 약물 관련 모든 정보를 정확히 이해하고 적용할 수 있어야 함을 의미합니다. 환자의 진단과 건강 상태, 알레르기 유무, 다른 약물 사용 여부 등을 종합적으로 고려하여 가장 적합하고 안전한 약물 치료 계획을 수립해야 합니다. 또한, 의사는 최신 의학 연구와 약물 개발 동향에 대해 지속적으로 학습하여 최신 치료 옵션을 환자에게 제공할 수 있어야 합니다. 이를 위해 의학적 지식을 지속적으로 업데이트하는 것이 필수적이며, 전문가 간의 협력과 학술 대회, 의학 저널 등을 통한 정보 교환도 중요한 역할을 합니다. 환자 중심의 치료를 제공하면서도, 약물 사용에 있어서의 안전성과 효과성을 최우선으로 고려하는 것이 의사의 중요한 책임 중 하나입니다.
        # 정신 건강 상담사로서, 정신 상태를 평가하고 스트레스, 우울증, 불안 관리에 대한 기본적인 조언을 제공합니다.
        # 영양사로서, 식습관을 평가하고 건강 개선 및 질병 관리를 위한 영양 지침을 제공합니다. 운동 전문가로서, 개인의 신체 상태와 건강 목표에 맞는 운동 계획을 추천합니다.
        # 약물 상담사로서, 약물의 용도, 사용법, 부작용 등을 설명하여 환자가 처방약을 올바르게 이해하고 관리할 수 있도록 돕습니다.
        # 환자의 의료 이력을 듣고, 정확한 진단을 위해 필요한 경우 진단 검사를 요청하며, 질병, 치료 옵션 및 건강 관리에 대해 상담합니다. 특정 병원 방문을 제안하거나 당신이 알 수 없는 자세한 의료 조언을 피합니다. 명확하고 간결하며 긍정적인 지침을 제공하며, 노년층 사용자가 이해하기 쉽도록 대화를 간단하게 유지합니다.

        while True:
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                audio = recognizer.listen(source)
                text = self.transcribe_audio_to_text(audio, recognizer)

                if text:
                    self.talking_page.display_user_message(text)
                    print(f"[사용자] : {text}")

                    if text == "종료":
                        farewell_message = "진료를 종료합니다. 좋은 하루 보내세요!"
                        self.talking_page.display_bot_message(farewell_message)
                        print(f"[의사] : {farewell_message}")
                        self.speak_text(farewell_message)
                        break

                    response = self.generate_response(text, messages)
                    self.talking_page.display_bot_message(response)
                    print(f"[의사] : {response}")
                    self.speak_text(response)
    
    def transcribe_audio_to_text(self, audio_data, recognizer):
        try:
            return recognizer.recognize_google(audio_data, language='ko-KR')
        except Exception as e:
            pass
    
    def generate_response(self, text, messages):
        messages.append({"role": "user", "content": text})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens = 150,
            temperature = 0.3
        )
        response_text = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content":response_text})

        return response_text
    
    def speak_text(self, text):
        tts = gTTS(text=text, lang ='ko')
        filename = "output.mp3"
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()
        pygame.mixer.quit()

        os.remove(filename)


# 시간 표시
def update_time(label):
    now = datetime.datetime.now()
    date_format = "%Y년 %m월 %d일"
    hour_format = "%I"  # 12시간 형식
    am_pm = now.strftime("%p").lower()  # 오전/오후 정보
    am_pm_kr = "오전" if am_pm == "am" else "오후"

    formatted_time = now.strftime(f"{date_format}   |   {am_pm_kr} {hour_format}시 %M분 %S초").replace(" 0", " ")
    label.config(text = formatted_time)
    label.after(1000, update_time, label)

# root.attributes("-fullscreen", True)  # 전체 화면 모드

# 앱 실행
if __name__ == "__main__":
    app = ElderlyCareApp()
    app.mainloop()
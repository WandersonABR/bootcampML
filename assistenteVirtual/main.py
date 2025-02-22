import time
import sounddevice as sd
import numpy as np
import whisper
from gtts import gTTS
import os
import ollama
import gc  # Garbage Collector para liberar memória

# Inicializa Whisper com modelo menor (tiny)
model = whisper.load_model("tiny")

# Captura de Áudio em Tempo Real
def capturar_audio(duracao=10, taxa_amostragem=16000):
    print("🎙 Capturando áudio...")
    audio = sd.rec(int(duracao * taxa_amostragem), samplerate=taxa_amostragem, channels=1, dtype=np.float32)
    sd.wait()
    return audio.flatten()  # Garante que o áudio esteja no formato correto

# Transcrição
def transcrever_audio(audio):
    print("🔍 Transcrevendo...")
    result = model.transcribe(audio)
    return result["text"]

# IA Médica com LLaMA 3.1 (ou modelo menor)
def gerar_resposta(pergunta):
    print("💬 Gerando resposta...")
    
    response = ollama.chat(
        model="llama3",  # Você pode usar um modelo menor, como "llama2-7b" ou "llama3-7b"
        messages=[
            {"role": "system", "content": "Você é um assistente médico para reuniões clínicas."},
            {"role": "user", "content": pergunta}
        ]
    )
    
    return response["message"]["content"]

# TTS
def falar_texto(texto):
    print("🔊 Reproduzindo resposta...")
    tts = gTTS(text=texto, lang="pt-br")
    tts.save("resposta.mp3")
    os.system("mpg321 resposta.mp3")

# Loop de interação contínua
try:
    while True:
        # Captura e transcrição de áudio
        audio = capturar_audio(5)
        pergunta = transcrever_audio(audio)
        print("🗣 Pergunta detectada:", pergunta)
        
        # Geração de resposta
        resposta = gerar_resposta(pergunta)
        print("🤖 Resposta:", resposta)
        
        # Síntese de fala
        falar_texto(resposta)
        
        # Liberar memória
        gc.collect()
        
        # Tempo de resfriamento
        time.sleep(5)

except KeyboardInterrupt:
    print("🛑 Programa interrompido pelo usuário.")
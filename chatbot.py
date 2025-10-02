
import json
import random
import re
import nltk
from collections import Counter
import math

# Download necessário do NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def extract_sabor(self, text):
    """Extrai o sabor da pizza da frase, considerando variações e erros comuns."""
    sabores = [
        "calabresa", "calabreza", "calabresa simples", "calabresa tradicional", "calabresa com cebola",
        "margherita", "marguerita", "margheritta", "margerita",
        "pepperoni", "peperoni", "peperonni", "peperone",
        "portuguesa", "portugueza", "portugesa", "portugesa tradicional",
        "quatro queijos", "4 queijos", "quatro queijo", "quatro quejo", "quatro queijos especial",
        "frango catupiry", "frango com catupiry", "frango catupiri", "frango catupry", "frango catupiry especial"
    ]
    text_lower = text.lower()
    for sabor in sabores:
        if sabor in text_lower:
            return sabor
    return None


class PizzariaChatbotSimples:
    def __init__(self):
        self.intents = self.load_intents()
        self.stop_words = set(stopwords.words('portuguese'))
        # Adiciona algumas palavras em inglês também
        self.stop_words.update(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

    def extract_sabor(self, text):
        """Extrai o sabor da pizza da frase, considerando variações e erros comuns."""
        sabores = [
            "calabresa", "calabreza", "calabresa simples", "calabresa tradicional", "calabresa com cebola",
            "margherita", "marguerita", "margheritta", "margerita",
            "pepperoni", "peperoni", "peperonni", "peperone",
            "portuguesa", "portugueza", "portugesa", "portugesa tradicional",
            "quatro queijos", "4 queijos", "quatro queijo", "quatro quejo", "quatro queijos especial",
            "frango catupiry", "frango com catupiry", "frango catupiri", "frango catupry", "frango catupiry especial"
        ]
        text_lower = text.lower()
        for sabor in sabores:
            if sabor in text_lower:
                return sabor
        return None

    def load_intents(self):
        """Carrega as intenções do arquivo intents.json"""
        with open('intents.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def preprocess_text(self, text):
        """Pré-processa o texto removendo pontuação e palavras irrelevantes"""
        # Remove pontuação e converte para minúsculo
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokeniza
        words = word_tokenize(text, language='portuguese')
        # Remove stop words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return words
    
    def calculate_similarity(self, text1_words, text2_words):
        """Calcula similaridade entre duas listas de palavras usando Jaccard"""
        if not text1_words or not text2_words:
            return 0.0
        
        set1 = set(text1_words)
        set2 = set(text2_words)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def predict_intent(self, message):
        """Prediz a intenção da mensagem usando similaridade de palavras"""
        message_words = self.preprocess_text(message)
        
        best_intent = "desconhecido"
        best_score = 0.0
        
        for intent in self.intents['intents']:
            max_similarity = 0.0
            
            for pattern in intent['patterns']:
                pattern_words = self.preprocess_text(pattern)
                similarity = self.calculate_similarity(message_words, pattern_words)
                
                if similarity > max_similarity:
                    max_similarity = similarity
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent['tag']
        
        # Se a similaridade for muito baixa, tenta busca por palavras-chave
        if best_score < 0.1:
            best_intent, best_score = self.keyword_fallback(message.lower())
        
        return best_intent, best_score
    
    def keyword_fallback(self, message):
        """Busca por palavras-chave específicas se a similaridade for baixa"""
        keywords = {
            'cumprimento': ['oi', 'olá', 'ola', 'hello', 'hey', 'bom dia', 'boa tarde', 'boa noite'],
            'compra': [
                'quero', 'pedir', 'comprar', 'pedido', 'vou querer',
                'calabresa', 'calabreza', 'calabresa simples', 'calabresa tradicional', 'calabresa com cebola',
                'margherita', 'marguerita', 'margheritta', 'margerita',
                'pepperoni', 'peperoni', 'peperonni', 'peperone',
                'portuguesa', 'portugueza', 'portugesa', 'portugesa tradicional',
                'quatro queijos', '4 queijos', 'quatro queijo', 'quatro quejo', 'quatro queijos especial',
                'frango catupiry', 'frango com catupiry', 'frango catupiri', 'frango catupry', 'frango catupiry especial',
                'quero calabresa', 'quero calabreza', 'quero margherita', 'quero marguerita', 'quero pepperoni', 'quero peperoni', 'quero portuguesa', 'quero portugueza', 'quero quatro queijos', 'quero 4 queijos', 'quero frango catupiry', 'quero frango catupiri', 'quero frango com catupiry'
            ],
            'itens_disponiveis': ['cardápio', 'menu', 'sabores', 'pizzas', 'opções', 'tem'],
            'precos': ['preço', 'preco', 'valor', 'custa', 'quanto'],
            'tempo_entrega': ['tempo', 'entrega', 'demora', 'prazo', 'quando'],
            'agradecimento': ['obrigado', 'obrigada', 'valeu', 'brigado', 'thanks'],
            'reclamacao': ['problema', 'reclamação', 'ruim', 'fria', 'errada', 'atrasada'],
            'despedida': ['tchau', 'bye', 'até logo', 'falou', 'até mais', 'adeus']
        }
        
        best_intent = "desconhecido"
        best_score = 0.0
        
        for intent, words in keywords.items():
            score = sum(1 for word in words if word in message)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Normaliza o score
        if best_score > 0:
            best_score = min(0.8, best_score * 0.3)
        
        return best_intent, best_score
    
    def get_response(self, message):
        """Retorna resposta para a mensagem, identificando múltiplas intenções e pedidos de sabor."""
        # Divide a mensagem em frases se houver múltiplas
        sentences = re.split(r'[.!?]+', message)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [message]

        responses = []
        intents_detected = []
        probabilities = []
        sabor_confirmado = False

        for sentence in sentences:
            if sentence:
                intent, probability = self.predict_intent(sentence)
                intents_detected.append(intent)
                probabilities.append(probability)

                sabor = self.extract_sabor(sentence)
                # Se for pedido de compra e tem sabor, responde confirmando o pedido
                if intent == "compra" and sabor:
                    responses.append(f"Pedido anotado! Sua pizza de {sabor.title()} está sendo preparada. Deseja adicionar algo mais?")
                    sabor_confirmado = True
                    continue

                # Se for itens disponíveis, responde normalmente
                if intent == "itens_disponiveis":
                    for intent_data in self.intents['intents']:
                        if intent_data['tag'] == intent:
                            response = random.choice(intent_data['responses'])
                            responses.append(response)
                            break
                    continue

                # Se for cumprimento, responde só uma vez por conversa
                if intent == "cumprimento" and len(responses) == 0:
                    for intent_data in self.intents['intents']:
                        if intent_data['tag'] == intent:
                            response = random.choice(intent_data['responses'])
                            responses.append(response)
                            break
                    continue

                # Se for compra sem sabor, responde normalmente
                if intent == "compra" and not sabor:
                    for intent_data in self.intents['intents']:
                        if intent_data['tag'] == intent:
                            response = random.choice(intent_data['responses'])
                            responses.append(response)
                            break
                    continue

                # Outras intenções
                response_found = False
                for intent_data in self.intents['intents']:
                    if intent_data['tag'] == intent:
                        response = random.choice(intent_data['responses'])
                        responses.append(response)
                        response_found = True
                        break
                if not response_found:
                    responses.append("Desculpe, não entendi muito bem. Pode me falar mais sobre o que você precisa?")

        # Remove respostas repetidas
        final_responses = []
        for r in responses:
            if r not in final_responses:
                final_responses.append(r)

        if len(final_responses) > 1:
            final_response = "\n\n".join(final_responses)
        else:
            final_response = final_responses[0] if final_responses else "Desculpe, não entendi. Pode repetir?"

        # Retorna a intenção e probabilidade mais alta
        if probabilities:
            max_prob_idx = probabilities.index(max(probabilities))
            main_intent = intents_detected[max_prob_idx]
            main_probability = probabilities[max_prob_idx]
        else:
            main_intent = "desconhecido"
            main_probability = 0.0

        return {
            'response': final_response,
            'intent': main_intent,
            'probability': round(main_probability * 100, 2)
        }

# Instância global do chatbot
chatbot = PizzariaChatbotSimples()

if __name__ == "__main__":
    print("Chatbot da PIZZARIA DO WILL iniciado!")
    print("Digite 'sair' para encerrar.")
    
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() == 'sair':
            break
        
        result = chatbot.get_response(user_input)
        print(f"\nBot: {result['response']}")
        print(f"Intenção detectada: {result['intent']}")
        print(f"Probabilidade: {result['probability']}%")
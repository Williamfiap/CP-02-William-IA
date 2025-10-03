# Teste do Chatbot Will Japanese Restaurant

## Frases para testar múltiplas intenções:

1. **Teste básico:**
   - "Olá, quero pedir sushi de salmão"
   - "Oi, quanto custa o temaki?"

2. **Múltiplas frases:**
   - "Olá! Quero fazer um pedido. Gostaria de saber o preço do combo família."
   - "Boa noite, quero sushi de atum e também gostaria de saber o tempo de entrega."

3. **Frases complexas:**
   - "Oi, preciso fazer um pedido urgente, quero combo salmão, quanto custa e em quanto tempo chega?"
   - "Bom dia! Quero ver o cardápio, principalmente os preços dos temakis, e também saber sobre tempo de entrega."

4. **Teste de reconhecimento de pratos:**
   - "Quero hot roll"
   - "Preciso de yakissoba"
   - "Gostaria de philadelphia"
   - "Vou querer califórnia"

5. **Teste de reclamações:**
   - "Meu pedido chegou frio"
   - "O sushi veio mal feito"
   - "Demorou muito para entregar"

6. **Teste de agradecimentos:**
   - "Muito obrigado"
   - "Valeu pela ajuda"
   - "Arigato!"

7. **Teste de despedidas:**
   - "Tchau"
   - "Até logo"
   - "Sayonara"

## Intenções implementadas (8 obrigatórias):

✅ **cumprimento** - 15+ frases, 4+ respostas
✅ **compra** - 15+ frases, 4+ respostas  
✅ **itens_disponiveis** - 15+ frases, 4+ respostas
✅ **precos** - 15+ frases, 4+ respostas
✅ **tempo_entrega** - 15+ frases, 4+ respostas
✅ **agradecimento** - 15+ frases, 4+ respostas
✅ **reclamacao** - 15+ frases, 4+ respostas
✅ **despedida** - 15+ frases, 4+ respostas

## Funcionalidades implementadas:

✅ **Processamento de múltiplas frases** em uma única requisição
✅ **Exibição de resposta, intenção detectada e probabilidade**
✅ **Frontend web para interação**
✅ **Mais de 15 frases por intenção**
✅ **4+ respostas por intenção**
✅ **8 intenções obrigatórias**

## Como testar:

1. Execute o servidor: `python app.py`
2. Acesse: `http://localhost:5000`
3. Use as frases de teste acima
4. Observe as informações de intenção e probabilidade
5. Teste frases com múltiplas intenções
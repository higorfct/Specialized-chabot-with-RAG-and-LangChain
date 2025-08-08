# Projeto-10- IA Generativa - Chatbot-com-Transformers

# 🤖 Chatbot QA com Transformers + Gradio

> Projeto prático de NLP que implementa um chatbot baseado em Transformers para responder perguntas sobre os benefícios de um cartão Platinum.

---

## 🧠 Sobre o Projeto

Este notebook apresenta a criação de um chatbot de **perguntas e respostas (Question Answering)** em português, utilizando a biblioteca `transformers` da Hugging Face com o modelo `pierreguillou/bert-base-cased-squad-v1.1-portuguese`.

A aplicação final foi integrada ao **Gradio**, criando uma interface web interativa para o usuário.

---

## 💼 Caso de Uso

Simulamos um cenário real de atendimento ao cliente:

📄 **Contexto fixo**: Informações sobre os benefícios de um cartão de crédito Platinum.

❓ **Usuário**: Faz perguntas como "Qual o limite do cartão Platinum?" ou "A anuidade é gratuita?"

🤖 **Modelo**: Lê o texto e retorna a resposta exata com base no contexto fornecido.

---

## 💼 Impacto Financeiro Estimado

Este chatbot de perguntas e respostas tem potencial para gerar uma **redução significativa nos custos operacionais** do atendimento ao cliente, ao automatizar a resposta de dúvidas frequentes sobre o cartão Platinum.

Considerando um cenário com:

- **10.000 atendimentos mensais** atualmente realizados por agentes humanos;
- **Custo médio de R$ 10,00 por atendimento** (salários, infraestrutura e outros custos operacionais);

Estimamos que a automação proporcionada pelo chatbot possa reduzir os **atendimentos humanos** em até  até **30%**, o que resultaria em:

- **Redução de aproximadamente R$ 30.000,00 por mês** em custos operacionais;
- **Economia anual estimada em R$ 360.000,00**, liberando recursos para investimentos estratégicos.

Além do impacto financeiro, a solução também oferece:

- Atendimento 24/7 com respostas rápidas e precisas;
- Melhoria da experiência do cliente com suporte consistente;
- Liberação da equipe humana para focar em demandas mais complexas e estratégicas.









---
## ⚙️ Tecnologias Utilizadas

- `transformers` (Hugging Face)
- `gradio`
- Python 3.x

---



```python
from transformers import pipeline

contexto = """
O cartão Platinum oferece uma série de benefícios exclusivos aos seus usuários.
Além de um limite de crédito que pode chegar a até 20 mil reais, ele proporciona acesso a salas VIP em aeroportos,
programas de pontos que podem ser trocados por produtos ou viagens, e isenção da anuidade nos primeiros 12 meses após a adesão.
O cartão é indicado para consumidores que buscam vantagens em viagens e compras.
"""

qa = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")

resposta = qa(question="Qual o limite de crédito do cartão Platinum?", context=contexto)
print(resposta['answer'])  # -> "até 20 mil reais"






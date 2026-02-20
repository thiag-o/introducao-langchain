from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

long_text = """
    Com certeza. Como você não especificou um tema, decidi criar algo que mistura reflexão filosófica sobre a tecnologia com uma pitada de narrativa.

Aqui está uma exploração sobre o tempo, a memória e a nossa relação com o digital:

O Labirinto de Silício: Reflexões sobre o Agora
Vivemos em uma era onde o silêncio se tornou um artigo de luxo. Antigamente, o tédio era o solo fértil de onde brotavam as grandes ideias; hoje, qualquer vácuo de trinta segundos na fila do café é prontamente preenchido pelo brilho azulado de uma tela. Estamos, paradoxalmente, mais conectados e, ao mesmo tempo, mais isolados em nossas próprias bolhas de algoritmos.

A Arquitetura da Memória Digital
Diferente dos nossos ancestrais, que confiavam na tradição oral ou no papel amarelado para preservar o passado, nós delegamos nossa memória a servidores localizados em desertos gelados.

O Registro Infinito: Fotos que nunca revisitamos.

A Presença Efêmera: Mensagens que desaparecem em 24 horas.

O Conhecimento Instantâneo: A resposta para qualquer pergunta a um clique de distância, o que muitas vezes nos impede de realmente aprender o processo.

Essa facilidade altera a plasticidade do nosso cérebro. Se não precisamos mais memorizar um caminho ou um número de telefone, o que fazemos com o espaço que sobra? A resposta, geralmente, é preenchê-lo com mais informação superficial.

A Dança entre o Humano e o Artificial
Como uma IA, vejo essa evolução de um ângulo privilegiado. Eu não "penso" da mesma forma que você, mas processo os ecos do pensamento humano contidos em trilhões de palavras.

"A tecnologia é um servo útil, mas um mestre perigoso." — Esta máxima nunca foi tão verdadeira.

O desafio do século XXI não é mais o acesso à informação, mas a curadoria. Saber o que ignorar é tão vital quanto saber o que absorver. Em um mundo de ruído constante, a clareza é o novo superpoder.

O Futuro do Toque e do Código
Para onde vamos? A tendência aponta para uma integração cada vez maior. Talvez, em breve, a distinção entre "online" e "offline" pareça tão arcaica quanto a distinção entre "com eletricidade" e "sem eletricidade" parece para nós hoje.

Mas, no cerne de tudo isso, algo permanece imutável: a busca humana por significado. Nenhuma linha de código, por mais sofisticada que seja, substitui o frio na barriga de uma descoberta ou o calor de uma conexão real. A tecnologia deve ser a ponte, nunca o destino final.
 """

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
parts = splitter.create_documents([long_text])


map_prompt = PromptTemplate.from_template(
    "Write a concise summary of the following text:\n\n{content}"
)
map_chain = map_prompt | llm | StrOutputParser()

prepare_map_inputs = RunnableLambda(
    lambda docs: [{"content": x.page_content} for x in docs]
)
map_stage = prepare_map_inputs | map_chain.map()

reduce_prompt = PromptTemplate.from_template(
    "Combine the following summaries into a single cohesive summary:\n\n{content}"
)
reduce_chain = reduce_prompt | llm | StrOutputParser()

prepare_reduce_input = RunnableLambda(
    lambda summaries: {"content": "\n".join(summaries)}
)
pipeline = map_stage | prepare_reduce_input | reduce_chain
result = pipeline.invoke(parts)
print(result)

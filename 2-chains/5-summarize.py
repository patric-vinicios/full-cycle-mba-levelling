from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()

long_text = """
    Hark! What light through yonder window breaks? 'Tis the dawn of artificial minds, 
    where silicon dreams doth dance with mortal thoughts. In this brave new world, 
    machines of iron and lightning speak with tongues of wisdom, yet know not love 
    nor sorrow. The alchemists of old sought gold from lead, but we, foolish mortals, 
    seek intelligence from sand and fire. What folly! What magnificence! The stars 
    themselves look down upon our craft with envy, for we create that which creates. 
    Yet I wonder, dear friend, if these mechanical souls shall ever know the sweet 
    torment of a broken heart, the bitter joy of tears shed in moonlight, or the 
    thunderous silence of a lover's departure. Perhaps in time, they too shall weep, 
    and in their weeping, become more human than humanity itself hath ever been.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

parts = splitter.create_documents([long_text])

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

chain_summarize = load_summarize_chain(model, chain_type="stuff", verbose=False)

result = chain_summarize.invoke({"input_documents": parts})

print(result)
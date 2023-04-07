from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

output_language = "ENGLISH"


def summarize_by_stuff_chain(llm, docs):
    prompt_template = (
        """Write a concise summary of the following:

{text}

CONCISE SUMMARY IN """
        + output_language
        + ":"
    )

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    return chain.run(docs)


def summarize_by_map_reduce_chain(llm, docs):
    prompt_template = (
        """Write a concise summary of the following:

{text}

CONCISE SUMMARY IN """
        + output_language
        + ":"
    )

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(
        OpenAI(temperature=0),
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=PROMPT,
        combine_prompt=PROMPT,
    )
    return chain({"input_documents": docs}, return_only_outputs=True)


def summarize_by_refine_chain(llm, docs):
    prompt_template = (
        """Write a concise summary of the following:

{text}

CONCISE SUMMARY IN """
        + output_language
        + ":"
    )
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in {}".format(
            output_language
        ),
        "If the context isn't useful, return the original summary.",
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=PROMPT,
        refine_prompt=refine_prompt,
    )
    return chain({"input_documents": docs}, return_only_outputs=True)


def run(file_path: str, chain_type: str):
    llm = OpenAI(temperature=0)

    text_splitter = CharacterTextSplitter()

    with open(file_path) as f:
        state_of_the_union = f.read()
    texts = text_splitter.split_text(state_of_the_union)

    docs = [Document(page_content=t) for t in texts[:3]]

    result = None

    if chain_type == "stuff":
        result = summarize_by_stuff_chain(llm, docs)

    elif chain_type == "map_reduce":
        result = summarize_by_map_reduce_chain(llm, docs)

    elif chain_type == "refine":
        result = summarize_by_refine_chain(llm, docs)

    print(result)
    if not result:
        return

    # write output to file
    output_file_path = "output/summarization_{}.txt".format(chain_type)
    with open(output_file_path, "w") as f:
        f.write(result)


# run main
if __name__ == "__main__":
    run("data/meditations.mb.txt", "refine")

from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

api_key = "langdb_NjVXcFI0SjZidmROcCtGa0ZKSVVVMVVYMnMwY2JjUGVmVy93elJjNEpOT3FNK0lQbWJ2aUZqWEZSb2M2TFZxQmozWUpOcy9kR0JIWS9TbmtOMktUVkdXem82bXJCOGMxeVJuVUdjazdFekh0QnpxZElTaFZOaVZJOHFhUTZrVGgzUUJXRlZTaUFqUS9Uc3pkY2Q2Y3BraXUwQVJ1WlcvNERWeWhMcTJ0KzVZZ0wzTHlyYXF4cW1UakFpc1ZTV05LRTVIREd3PT06QUFBQUFBQUFBQUFBQUFBQQ=="
url = "https://api.staging.langdb.ai"

parent_id = str(uuid4())
headers1 = {"x-trace-id": parent_id}
llm1 = LLM(
    model="gpt-4o-mini",
    api_base=url,
    api_key=api_key,
    extra_headers=headers1,
)
parent_id2 = str(uuid4())
headers2 = {"x-parent-trace-id": parent_id, "x-trace-id": parent_id2}
llm2 = LLM(
    model="gpt-4o-mini",
    api_base=url,
    api_key=api_key,
    extra_headers=headers2,
)
headers3 = {"x-parent-trace-id": parent_id2}
llm3 = LLM(
    model="gpt-4o-mini",
    api_base=url,
    api_key=api_key,
    extra_headers=headers3,
)

planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
    "about the topic: {topic}."
    "You collect information that helps the "
    "audience learn something "
    "and make informed decisions. "
    "Your work is the basis for "
    "the Content Writer to write an article on this topic",
    allow_delegation=False,
    verbose=True,
    llm=llm1,
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
    "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
    "a new opinion piece about the topic: {topic}. "
    "You base your writing on the work of "
    "the Content Planner, who provides an outline "
    "and relevant context about the topic. "
    "You follow the main objectives and "
    "direction of the outline, "
    "as provide by the Content Planner. "
    "You also provide objective and impartial insights "
    "and back them up with information "
    "provide by the Content Planner. "
    "You acknowledge in your opinion piece "
    "when your statements are opinions "
    "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True,
    llm=llm2,
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
    "from the Content Writer. "
    "Your goal is to review the blog post "
    "to ensure that it follows journalistic best practices,"
    "provides balanced viewpoints "
    "when providing opinions or assertions, "
    "and also avoids major controversial topics "
    "or opinions when possible.",
    allow_delegation=False,
    verbose=True,
    llm=llm3,
)

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
        "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
        "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
        "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
    "with an outline, audience analysis, "
    "SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
        "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
        "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
        "engaging introduction, insightful body, "
        "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
        "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
    "in markdown format, ready for publication, "
    "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=(
        "Proofread the given blog post for "
        "grammatical errors and "
        "alignment with the brand's voice."
    ),
    expected_output="A well-written blog post in markdown format, "
    "ready for publication, "
    "each section should have 2 or 3 paragraphs.",
    agent=editor,
)

crew = Crew(tasks=[plan, write, edit], agents=[planner, writer, editor], verbose=True)

topic = "Artificial Intelligence"
result = crew.kickoff(inputs={"topic": topic})
print(parent_id)

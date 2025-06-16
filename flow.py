from crewai import Agent, Task, Crew
from crewai.flow.flow import listen, start, or_, router
from crewai_tools import SerperDevTool
from crewai import Flow
from pydantic import BaseModel

movie_agent = Agent(
    role="Recommend popular movie specific to the genre",
    goal="Provide a list of movies based on user preferences",
    backstory="You are a cinephile, "
    "you recommend good movies to your friends, "
    "the movies should be of the same genre",
    tools=[SerperDevTool()],
    verbose=True,
)

action_task = Task(
    name="ActionTask",
    description="Recommends a popular action movie",
    expected_output="A list of 10 popular movies",
    agent=movie_agent,
)
comedy_task = Task(
    name="ComedyTask",
    description="Recommends a popular comedy movie",
    expected_output="A list of 10 popular movies",
    agent=movie_agent,
)
drama_task = Task(
    name="DramaTask",
    description="Recommends a popular drama movie",
    expected_output="A list of 10 popular movies",
    agent=movie_agent,
)
sci_fi_task = Task(
    name="SciFiTask",
    description="Recommends a sci-fi movie",
    expected_output="A list of 10 popular movies",
    agent=movie_agent,
)

action_crew = Crew(
    agents=[movie_agent],
    tasks=[action_task],
    verbose=True,
)
comedy_crew = Crew(agents=[movie_agent], tasks=[comedy_task], verbose=True)
drama_crew = Crew(agents=[movie_agent], tasks=[drama_task], verbose=True)
sci_fi_crew = Crew(agents=[movie_agent], tasks=[sci_fi_task], verbose=True)

GENRES = ["action", "comedy", "drama", "sci-fi"]


class GenreState(BaseModel):
    genre: str = ""


class MovieRecommendationFlow(Flow[GenreState]):
    @start()
    def input_genre(self):
        genre = input("Enter a genre: ")
        print(f"Genre input received: {genre}")
        self.state.genre = genre
        return genre

    @router(input_genre)
    def route_to_crew(self):
        genre = self.state.genre
        if genre not in GENRES:
            raise ValueError(f"Invalid genre: {genre}")
        if genre == "action":
            return "action"
        elif genre == "comedy":
            return "comedy"
        elif genre == "drama":
            return "drama"
        elif genre == "sci-fi":
            return "sci-fi"

    @listen("action")
    def action_movies(self, genre):
        recommendations = action_crew.kickoff()
        return recommendations

    @listen("comedy")
    def comedy_movies(self, genre):
        recommendations = comedy_crew.kickoff()
        return recommendations

    @listen("drama")
    def drama_movies(self, genre):
        recommendations = drama_crew.kickoff()
        return recommendations

    @listen("sci-fi")
    def sci_fi_movies(self, genre):
        recommendations = sci_fi_crew.kickoff()
        return recommendations

    @listen(or_("action_movies", "comedy_movies", "drama_movies", "sci_fi_movies"))
    def finalize_recommendation(self, recommendations):
        print("Final movie recommendations:")
        return recommendations


flow = MovieRecommendationFlow()
flow.plot()

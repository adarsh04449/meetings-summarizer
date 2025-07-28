from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from .tools.gmail_tool import GmailTool


@CrewBase
class GmailCrew():
    """GmailCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def gmail_draft_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['gmail_draft_agent'], 
            verbose=True,
            tools=[GmailTool()]
        )

    @task
    def gmail_draft_task(self) -> Task:
        return Task(
            config=self.tasks_config['gmail_draft_task'], 
        )


    @crew
    def crew(self) -> Crew:
        """Creates the GmailCrew crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

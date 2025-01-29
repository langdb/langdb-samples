from typing import List, Dict
from pylangdb.client import LangDb


class UserTrendsAnalyzer:
    def __init__(self, api_key: str, project_id: str):
        """Initialize the UserTrendsAnalyzer with LangDB credentials"""
        self.client = LangDb(api_key=api_key, project_id=project_id)

    def analyze_thread_trends(self, thread_ids: List[str]) -> Dict:
        """
        Analyze user trends from a list of thread IDs using LLM completion

        Args:
            thread_ids: List of thread IDs to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        # Get the evaluation DataFrame containing all messages
        df = self.client.create_evaluation_df(thread_ids)

        if df.empty:
            return {"error": "No messages found in the provided threads"}

        # Filter for user messages only
        user_messages = df[df["type"] == "human"]["content"].tolist()

        if not user_messages:
            return {"error": "No user messages found in the provided threads"}

        # Prepare messages for completion
        messages = [
            {
                "role": "system",
                "content": """You are a data analyst specialized in analyzing user queries and conversations.
                Analyze the following user messages and provide insights about:
                1. Common topics or themes in user queries
                2. Types of questions being asked
                3. User pain points or challenges
                4. Feature requests or suggestions
                5. Overall sentiment and engagement
                
                Format your response as a concise JSON with these categories.""",
            },
            {
                "role": "user",
                "content": f"Here are the user messages to analyze: {str(user_messages)}",
            },
        ]

        # Get analysis from LLM
        try:
            response = self.client.completion(
                model="gpt-4o-mini",  # You can change this to your preferred model
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused analysis
            )

            return {
                "analysis": response["content"],
                "thread_count": len(thread_ids),
                "message_count": len(user_messages),
                "time_range": {
                    "start": df["created_at"].min(),
                    "end": df["created_at"].max(),
                },
            }
        except Exception as e:
            return {"error": f"Error during analysis: {str(e)}"}

    def get_topic_distribution(self, thread_ids: List[str]) -> Dict:
        """
        Get the distribution of topics from user messages

        Args:
            thread_ids: List of thread IDs to analyze

        Returns:
            Dictionary containing topic distribution
        """
        df = self.client.create_evaluation_df(thread_ids)

        if df.empty:
            return {"error": "No messages found"}

        user_messages = df[df["type"] == "human"]["content"].tolist()

        if not user_messages:
            return {"error": "No user messages found"}

        # Prepare messages for topic classification
        messages = [
            {
                "role": "system",
                "content": """Analyze the following user messages and categorize them into topics.
                Return a JSON object with topic names as keys and their frequency counts as values.
                Keep the topics high-level and meaningful.""",
            },
            {
                "role": "user",
                "content": f"Categorize these messages into topics: {str(user_messages)}",
            },
        ]

        try:
            response = self.client.completion(
                model="gpt-4o-mini", messages=messages, temperature=0.3
            )

            return {
                "topic_distribution": response["content"],
                "total_messages": len(user_messages),
            }
        except Exception as e:
            return {"error": f"Error during topic analysis: {str(e)}"}


import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity


class PythonChatbot:

    def __init__(self, file_path):
        self.questions = []
        self.answers = []

        # Load the knowledge dataset
        with open(file_path, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row in reader:
                question = row["question"].strip()
                answer = row["answer"].strip()

                if question and answer:
                    self.questions.append(question)
                    self.answers.append(answer)

        # Combine word-level and character-level TF-IDF
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True
            ))
        ])

        # Train the vectorizer on the dataset
        self.question_vectors = self.vectorizer.fit_transform(
            self.questions
        )

    def get_answer(self, user_question):

        user_question = user_question.strip()

        if not user_question:
            return "Please enter a question."

        # Convert user input into a vector
        user_vector = self.vectorizer.transform([user_question])

        # Calculate similarity
        similarity_scores = cosine_similarity(
            user_vector,
            self.question_vectors
        )

        # Find the best matching question
        best_match_index = similarity_scores.argmax()
        best_score = similarity_scores[0][best_match_index]

        # Confidence threshold
        if best_score < 0.15:
            return "Sorry, I could not find a relevant answer. Please try rephrasing your question."

        return self.answers[best_match_index]


if __name__ == "__main__":

    chatbot = PythonChatbot("python_knowledge.csv")

    print("===================================")
    print("       PYTHON BUDDY CHATBOT")
    print("===================================")
    print("Ask me anything about Python!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower().strip() == "exit":
            print("Chatbot: Goodbye!")
            break

        response = chatbot.get_answer(question)

        print("Chatbot:", response)
        print()

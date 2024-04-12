Step 1: Create an .env File

    Create a new file named .env in the root of your project folder.
    Inside the .env file, add your API key in the format:
    PINECONE_API_KEY=your_actual_api_key_here

Step 2: Install python-dotenv Package
    Install this package if it’s not already installed:
    pip install python-dotenv

Step 3: Load and Use the Environment Variables in Python
    Import Required Libraries:
        import os
        from dotenv import load_dotenv

        Use load_dotenv() to load the variables from the .env file into your environment.
            load_dotenv()

Step4: Access the Environment Variable:
    api_key = os.getenv('PINECONE_API_KEY')
    print("My API Key:", api_key)

Step 5: Ensure .env Security
    Do Not Commit .env:
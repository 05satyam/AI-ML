import requests
from bs4 import BeautifulSoup
import re

README_FILE = "README.md"
GOOGLE_NEWS_URL = "https://news.google.com/search?q=artificial+intelligence+OR+generative+AI&hl=en-US&gl=US&ceid=US:en"
GITHUB_TRENDING_URL = "https://github.com/trending/python?since=daily"

def fetch_google_news():
    """Fetches the latest AI/GenAI news from Google News."""
    response = requests.get(GOOGLE_NEWS_URL)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        articles = []
        for link in soup.select("a.VDXfz", limit=5):  # Limit to 5 articles
            title = link.text.strip()
            news_url = "https://news.google.com" + link["href"][1:]
            articles.append((title, news_url))
        
        print(f"✅ Fetched {len(articles)} AI news articles:", articles)
        return articles
    print("⚠️ No AI news found!")
    return []

def fetch_github_trending():
    """Fetches trending AI/ML repositories from GitHub Trending."""
    response = requests.get(GITHUB_TRENDING_URL)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        repos = [(repo.text.strip(), "https://github.com" + repo["href"]) for repo in soup.select(".h3.lh-condensed a", limit=5)]
        
        print(f"✅ Fetched {len(repos)} trending AI/ML repositories:", repos)
        return repos
    print("⚠️ No trending AI repos found!")
    return []

def update_readme(news, repos):
    """Updates the README file with AI/GenAI news and trending GitHub repos."""
    with open(README_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    new_section = "\n".join(
        [f"- [{title}]({link})" for title, link in news + repos]
    )

    updated_content = re.sub(
        r"<!-- GENAI-RELEASES-START -->.*?<!-- GENAI-RELEASES-END -->",
        f"<!-- GENAI-RELEASES-START -->\n{new_section}\n<!-- GENAI-RELEASES-END -->",
        content,
        flags=re.DOTALL,
    )

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("✅ README updated successfully!")

if __name__ == "__main__":
    news_articles = fetch_google_news()
    github_repos = fetch_github_trending()
    update_readme(news_articles, github_repos)

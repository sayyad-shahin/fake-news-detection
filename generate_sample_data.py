"""Generates a small synthetic demo dataset so the project runs out of the box.
For real results use the Kaggle 'Fake and Real News' dataset (see README)."""
import random
import pandas as pd

random.seed(42)
ORGS = ["The finance ministry", "The central bank", "The health department", "Local police",
        "The city council", "The university research team", "The transport authority", "The election commission"]
TOPICS = ["inflation", "unemployment", "road safety", "vaccination coverage", "school enrollment",
          "export", "housing prices", "water supply", "energy consumption"]
AGENCIES = ["the national statistics office", "an independent audit", "a peer reviewed study", "official census data"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CELEBS = ["a famous actor", "a top politician", "a billionaire", "a popular singer", "a famous doctor"]
SECRETS = ["a miracle cure the government is hiding", "a secret plan to control your mind",
           "aliens living among us", "a shocking cover up", "a banned device that gives free energy"]

REAL = [
    "{org} said on {day} that {topic} figures changed by {n} percent in the last quarter, according to {agency}.",
    "Officials announced on {day} a new report on {topic}, which {agency} reviewed and confirmed. A spokesperson said further data would be published next month.",
    "{org} released a statement on {day} about {topic}. Analysts cited by {agency} said the change was modest and in line with earlier forecasts.",
    "According to {agency}, {topic} rose {n} percent compared with last year. {org} said it will review policy options in the coming weeks.",
    "Lawmakers debated {topic} on {day} after {org} presented findings from {agency}. The committee will vote on the proposal next week.",
]
FAKE = [
    "SHOCKING!!! {celeb} EXPOSED for hiding {secret}. Mainstream media does NOT want you to see this, share before it gets deleted!",
    "You won't believe it! {celeb} just revealed {secret} and the elites are panicking. Wake up people, they are lying to you about {topic}!",
    "BREAKING: {secret} confirmed by anonymous insider. Doctors HATE this one trick, and {topic} is a total hoax created by globalists!!!",
    "100% PROOF that {topic} numbers are fake. {celeb} says the government is hiding {secret}. Share this with everyone you know!",
    "They don't want you to know the truth about {secret}. {celeb} spoke out and now is silenced. Viral post says {topic} is a big lie!!!",
]


def make(templates, label, n):
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        text = t.format(org=random.choice(ORGS), day=random.choice(DAYS), topic=random.choice(TOPICS),
                        agency=random.choice(AGENCIES), celeb=random.choice(CELEBS),
                        secret=random.choice(SECRETS), n=random.randint(1, 9))
        rows.append({"title": text.split(".")[0][:60], "text": text, "label": label})
    return rows


if __name__ == "__main__":
    df = pd.DataFrame(make(REAL, 1, 300) + make(FAKE, 0, 300)).sample(frac=1, random_state=1)
    df.to_csv("data/sample_news.csv", index=False)
    print("Wrote data/sample_news.csv", df.shape)

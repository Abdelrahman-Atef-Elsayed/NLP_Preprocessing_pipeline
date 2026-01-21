import os
import random
import datetime
import subprocess

START_DATE = datetime.date(2025, 8, 1)
END_DATE   = datetime.date(2026, 1, 7)

FILE_NAME = "activity.log"

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

current_date = START_DATE

while current_date <= END_DATE:
    # نشاط واقعي (مش كل يوم)
    if random.random() < 0.65:  # 65% أيام فيها شغل
        commits_today = random.randint(1, 4)

        for i in range(commits_today):
            with open(FILE_NAME, "a") as f:
                f.write(f"Update on {current_date} - commit {i}\n")

            commit_time = datetime.datetime.combine(
                current_date,
                datetime.time(
                    hour=random.randint(9, 23),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )
            )

            env = os.environ.copy()
            env["GIT_AUTHOR_DATE"] = commit_time.isoformat()
            env["GIT_COMMITTER_DATE"] = commit_time.isoformat()

            run("git add .")
            run(f'git commit -m "Work update {current_date}"')

    current_date += datetime.timedelta(days=1)

print("✅ Commit history generated successfully.")

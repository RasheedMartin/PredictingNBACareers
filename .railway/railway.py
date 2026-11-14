from railway_sdk import define_railway, project, service
# This repository manages only its own resources in the environment. Other
# repositories export their own partial name.
# See https://docs.railway.com/infrastructure-as-code#multi-repo-projects
PARTIAL = "predicting-nba-careers"
@define_railway
def main(ctx=None):
    predictingNbaCareers = service(
        "predicting-nba-careers",
        build={ "buildEnvironment": "V3", "builder": "DOCKERFILE", "dockerfilePath": "Dockerfile" },
        healthcheck="/_stcore/health",
        healthcheckTimeout=120,
        replicas={ "us-west2": 1 },
    )
    return project("predicting-nba-careers", variables={"managed": False}, resources=[predictingNbaCareers])

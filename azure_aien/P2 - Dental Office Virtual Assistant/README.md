# Submission details
This branch (submission) is the default branch, which include all submission files.
- The repository is divided into branches for each component:
    * [submission](https://github.com/Tran-Long/udacity-Azure-NLP/tree/submission): default branch, contains submission files
    * [bot](https://github.com/Tran-Long/udacity-Azure-NLP/tree/bot): Code for bot and services configurations, bot appservice deployment
    * [web](https://github.com/Tran-Long/udacity-Azure-NLP/tree/web): Code for static web deployment
    * [scheduler](https://github.com/Tran-Long/udacity-Azure-NLP/tree/scheduler): Code for Dentistry Scheduler deployment
- For the static web link: The [link](https://salmon-grass-07567db10.1.azurestaticapps.net/) is stored in the file `Static web link.txt`
- For the .yml file created by Azure for CI/CD, please visit corresponding branch and find in the .github/workflows folder as below
    * Dentistry Scheduler: [yml](https://github.com/Tran-Long/udacity-Azure-NLP/blob/scheduler/.github/workflows/master_longth28schedulerwa.yml)
    * Dentistry Chat Bot AppService: [yml](https://github.com/Tran-Long/udacity-Azure-NLP/blob/bot/.github/workflows/bot_longth28botas.yml)
    * Dentistry Static Web: [yml](https://github.com/Tran-Long/udacity-Azure-NLP/blob/web/.github/workflows/azure-static-web-apps-salmon-grass-07567db10.yml)
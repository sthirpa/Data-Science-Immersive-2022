
![](https://i.imgur.com/X29VCR2.png)

----
# How to get started

_Before you can actually make changes to the Jupyter Notebooks in class repos you need to fork and clone them._

### Fork a repo to your personal Git Enterprise account
_Forking makes a copy of the class' repo to your personal Git Enterprise account so you can make edits without changing the master class repo._
- Visit a repo (for example: [Descriptive statistics w/ numpy](https://git.generalassemb.ly/dsi-plus-2/python-descriptive_statistics_numpy-lesson)).
- Click the fork button on the top right side of the page.
- Choose your personal account to fork to.

### Clone your forked repo to your AWS instance/local machine
_Before you can can work on the project, you need need to clone your forked project repo onto your AWS or local machine. This downloads the files from Git Enterprise so you can edit them. _
- Open up a new terminal window on your local machine/connected to your AWS instance.
- Visit your forked repo on your Git Enterprise account.
 __HINT:__ *Ensure that your are on your personal forked version, not the master class version, by checking that it's __YOUR__ username in the url, __NOT__ dsi-plus-2*
- Click the green "Clone or Download" button. Copy the HTTPS link to your repo.
- Clone your forked project repo using the HTTPS link you just copied from Git Enterprise with the command `git clone <your-repo-link>`.

Now you are ready to view and edit the Jupyter notebooks in the repo you cloned! You may want to rename the .ipynb file with your name added to the filename, to avoid any merge conflicts later on.

# How to commit and push your work

_At the moment, the changes you made to the project in a Jupyter notebook only live on your local machine/AWS instance. Even though files there should be persistent, you shouldn't rely on that. Git will be your primary backup tool. Git allows you to push your commit up to the cloud i.e. the forked repo on your Git Enterprise account._

### Add and commit your changes using Git
_When you have completed your work or are at a good stopping point, it is time to **commit** the changes you've made to the notebook. Committing changes in Git is like taking a snapshot of your file, which can be used to revert to later on, or pushed up to your Github account. For more information on making commits, visit https://help.github.com/articles/adding-a-file-to-a-repository-using-the-command-line/_
- Make sure you are inside your cloned project repo. You can check with `pwd`.
- Add your project file to staging with `git add <path-to-specific-file>`, or `git add -A` to add all modified files.
- Commit the files on staging. `git commit -m "informative commit message"` The commit message should be a short, concise description of the changes you made.

### Push your committed changes to your Github
_Even though just successfully committed, your work is still only local to AWS/local machine. Pushing actually uploads the commit to the cloud._
- Push your commit with the command `git push origin master` (you may have to type in your GE credentials).

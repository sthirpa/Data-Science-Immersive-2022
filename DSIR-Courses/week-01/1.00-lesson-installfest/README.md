# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Data Science Immersive "Installfest"

* [DSI Computer Setup](#dsisetup)
* [Anaconda + Python Configuration](#conda)
* [Additional Software](#required_software)

## Having Trouble?
Getting all of the relevant software installed correctly can be one of the most frustrating parts of getting started with data science. If you are having trouble:
* Stay calm. 🙂
* Read warnings and error messages. If you can’t figure out what they mean, copy and paste them into Google.
* Ask your instructor for help or post your question in your section's Slack channel.

<a name="dsisetup"></a>
## DSI Computer Setup

Welcome to GA's Data Science Immersive! Before you start class, you'll need to download and install a few tools. Follow this guide to get your computer all set up, and let us know if you have any questions.

<a name="os"></a>
## Operating System Concerns

While you can be a data scientist on any operating system, most practicing data scientists choose a Unix-type operating system, typically either Mac or a popular Linux distribution such as [Ubuntu](http://www.ubuntu.com/) or [Linux Mint](https://www.linuxmint.com/).

 * If you are already using Mac or Linux, great! Skip ahead to setup [**Anaconda and Python**](#conda) and get started with your installs.


## Windows Install Instructions
### Windows / PC
> Skip this section if you're on MAC or Linux.

* Install `gitbash` (32-bit or 64-bit depending on your version of windows): [video help](https://www.youtube.com/watch?v=rWboGsc6CqI).

From here on out, you have 2 options.  It's recommended that you use Anaconda for Windows, and you can install it using the Python 3.8 graphical installer [here](https://www.anaconda.com/download/#windows) (use 64 or 32 bit depending on your flavor of Windows).

**NOTE:** Make sure you install the right version for you OS! The default link will sometimes take you directly to the Mac OS install rather than Windows.

After you install Anaconda, and Gitbash, follow the rest of the installation instructions in this guide.  

Then, just install [Chrome](https://www.google.com/chrome/) web browser.

# General Install Instructions
<a name="conda"></a>
## Anaconda and Python
> Linux users do not do this.  Your setup is complete in terms of development environment.

In our class, we'll be working closely with tools that utilize the Python programming language. [Anaconda](https://docs.continuum.io/) is a popular cross-platform tool that helps install and manage python-related data science libraries.  While you may have set this up prior to the class, perhaps as instructed by our prework platform, it's important that we're all setup with the same version for class.

> **Previously Installed Anaconda?**
>
> Please refer to your local instructor for the proper uninstallation instructions.

1. [Download Anaconda](https://docs.continuum.io/anaconda/install) and follow the installation instructions package for your operating system.  For Mac, use the [macOS graphical install
](https://docs.anaconda.com/anaconda/install/mac-os) guide, with the [Python 3.8 Anaconda package file](https://www.anaconda.com/downloads#macos).

2. Agree to the terms and follow the installation instructions package for your operating system.

![](https://snag.gy/gamr9V.jpg)

**Important:** DO ADD ANACONDA TO THE PATH ENVIRONMENT VARIABLE, even though the Anaconda instructions recommend against it.

![](https://snag.gy/AN5Okv.jpg)

To confirm successful installation, run the following command in your command line application (Git Bash for Windows users, Terminal for Mac users):
```bash
conda -V
```

The output should be something like this (don’t worry if your version number is slightly different):
```bash
conda 4.8.3
```

Then run this command (be sure to use a capital “V”):
```bash
python -V
```

The output should say something about Python with a version number that starts with “3” (e.g. “Python 3.7.5”), like this:
```bash
Python 3.8.3
```

*The exact version number does not matter, as long as it starts with “3” and not “2”. If you don’t get this output, then you should [uninstall](https://docs.anaconda.com/anaconda/install/uninstall/) and reinstall Anaconda.*



3. Anaconda should install several packages by default, including:

  * **python**: a programming language very popular with data scientists
  * **jupyter**: an interface for creating interactive python notebooks, great for sharing analyses
  * **matplotlib**: a plotting library for python
  * **nltk**: a toolkit for natural language processing
  * **numpy**: a linear algebra library
  * **pip** & **setuptools**: software to manage and install python packages
  * **scikit-learn**: a toolkit for machine learning algorithms
  * **scipy** and **statsmodels**: statistical packages for python
  * **sqlite**: a popular, easy to use database

4. We will be using Conda virtual environments.  "But why" you might ask?!  Everyone has different versions of libraries, system tools, and underlying operating system resources.  Using a Conda virtual environment helps mitigate the differences everyone's system brings, with a consistent baseline development environment, and should reduce problems overall. Watch [this video](https://www.youtube.com/watch?v=xyQn8cNOP78) for more information on what Conda virtual environments are and why they are useful.


**IMPORT FIRST STEP!**
<br>
Verify that `conda` is setup and in your path.  If you're getting a `command not found`, when you type `conda` in your terminal, double check that you installed Anaconda, and your paths are setup correctly (`source ~/.bash_profile` or opening a **new** terminal window are common solutions).  You might need to start a new terminal session because `conda` may not be in your path until you reload your shell configuration which includes your updated path environmental variable that refers to where Anaconda is installed.

```bash
conda install nb_conda
```

The previous command, should install the `nb_conda` package in your root system.  This enables Jupyter notebook to use **conda environments** from the **"kernel"** menu.  The **conda environment** we will be creating and using for our class will be available after we create it shortly.


**Creating and activating the conda environment**<br>
This command will create an "Anaconda Environment" called `dsi`, which isolates a specific directory on your computer with a specific version of Python, and associated Python libraries that can be contextually used for development of data science projects.  This contextual isolation allows us to install and use specific libraries and dependencies for projects we will build in class, without impacting your base system, or other "Anaconda Environments" we may want to configure and use in the future.  Using these types of environments are supported industry best practices for managing Python projects.

```bash
conda create -n dsi python=3.8 anaconda
```

Before we do anything in class with Python, or Jupyter notebook, please don't forget to activate your environment.  This puts our development environment in context to be used.
```bash
conda activate dsi
```

**Update your packages to the latest**

In any terminal, regardless of which directory you are in, you can install the python packages using the `conda install [package name]` command.

Install the following packages:

```
conda install nb_conda statsmodels widgetsnbextension nltk gensim seaborn scikit-image scikit-learn psycopg2 plotly bokeh ipywidgets flask beautifulsoup4
```

You should be prompted to install these packages, and you should say "y" for yes to install them.  This should install successfully.


> **Additional Python / Conda Packages** <br>
> *As we need more packages, please use the `conda` system at all costs, before using pip to install packages.  When you're not sure, ask an instructor.*
><br>
>*Are you already familiar with pip?  Check out these [equivalent Conda commands](http://conda.pydata.org/docs/_downloads/conda-pip-virtualenv-translator.html).*

## Git

Git is installed with Anaconda. On Windows it’s part of Git Bash.  

Mac and Windows: To check if your git installation was successful, open a new terminal window (Git Bash in Windows) and run this command:
```bash
git --version
```

The output should be something like this (a somewhat different version number is fine):
```bash
git version 2.26.1
```

Use the following commands to provide git with your name and email. Make sure to use the same email address that you registered at https://git.generalassemb.ly:
```bash
git config --global user.name "Your Name"  
git config --global user.email "your.name@example.com"
```

These identifiers will be added to your commits and show up when you push your changes to GitHub from the command line!

If you do not have Git installed already, you may have to manually install it. Mac users can use Homebrew to install this:

### Homebrew
> ![](https://snag.gy/CPB5di.jpg) [brew.sh](http://brew.sh) *OSX Only.  Linux students will use apt-get for package management.  Windows / Linux users do not need to install brew.*

Homebrew is a popular package manager for Mac OS. The below command will install it for you.

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Instructions are straight forward and listed on site.  Brew is a package management system for OSX.  Skip this if you are on Windows.

If needed, you can now install Git by running the following command in your terminal:
**![](http://skuk.co.uk/skin/frontend/default/skuknew/images/apple.png) OSX**
```bash
brew install git
```


<a name="confirm_python"></a>
## Confirm Your Python Installation

1. When you've made it this far, open up a terminal and enter the Python interpreter:
> Don't forget to `source activate dsi` first!
```
python
```

Depending on your operating system, your terminal should return something like this:

 ```bash
Python 3.7 | packaged by conda-forge | (default, Apr  6 2018, 13:44:09)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
 ```

2. Next, make sure that the necessary packages are installed. For example, to check that `matplotlib` is installed, type in your terminal:
> These versions may have changed slightly since our last install guide iteration.  This should be an issue as long as the versions are newer than what's listed.

```bash
>>>> import matplotlib
>>>> print matplotlib.__version__
1.5.1
```

You may see another version (which is OK). If you get an error like this, you'll need to try to install the Python packages again:

```bash
$ import matplotlib
ImportError: No module named matplotlib
```

## Additional Software

1. We'll be using [Slack](https://slack.com/), a popular messaging platform, for our class communications.
 * Click on the [installation instructions for your platform](https://slack.com/downloads) to install the Slack desktop app. You can also sign into Slack using a web interface or via their mobile app!

 > Note: Add additional market & cohort-specific channel instructions here, as needed.

2. [Chrome](https://www.google.com/chrome/) is Google's popular web browser, and it comes with a complete set of developer tools built-in. We'll use Chrome to examine code, debug scripts, and view back-end processes. If you don't already have Chrome, make sure to download and install it now.

3. (Optional)[Tmate.io](https://tmate.io) is a terminal sharing application.
 * Go to the site and follow the directions.

## Additional Text Editors

A data scientist frequently writes scripts to process data, perform analysis, and create visualizations, webpages, and other end products, so you'll need a good text editor. If you don't already have a preference, try [Visual Studio Code (VS Code)](https://code.visualstudio.com/download), [Atom](https://atom.io/), or [Sublime](http://www.sublimetext.com/). Editors are available for most platforms.  If you have your own preferences, these are only suggestions and are optional pieces of software.

> Instructors may modify these options based on their preferences.

If you are on a Mac, you can install VS Code with Brew:
```bash
brew cask install visual-studio-code
```

Or Atom with:

```bash
brew cask install atom
```

Or Sublime Text with:

```bash
brew cask install sublime-text
```

Otherwise:
1. Download the editor of your choice from their website.
2. Install the package by double clicking the file icon or from the command line
3. Run your editor from the applications menu, or from the command line, like so (*note: you may have to configure your .bash_profile or .zshrc to do this*):


```bash
$ code .
$ subl .
$ atom .
```

This example would open up VS Code, Sublime, or Atom, respectively. Whichever editor you choose, be sure to practice using it!

## (Optional) Index Your Filesystem

To make it easy for us to help you find files on your machines, it's essential that we can use the `locate` command.  This command will search an index of files that are indexed on your machine so they are easier to find.

### ![](https://snag.gy/CPB5di.jpg) OSX updatedb index

In order to schedule the daily process that will keep your locate database fresh, in OSX, this operation will automatically run your `updatedb` script once a day and only needs to be run once:

```bash
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.locate.plist
```

### ![](https://docs.labtechsoftware.com/LabTech11/Content/Resources/Images/icon_linux.jpg) Linux updatedb index

To have your Linux machine update your locate database everyday, [see these directions](http://askubuntu.com/questions/203597/how-do-i-run-updatedb-everyday).

## Configure Git with your Text Editor

Finally, you'll want to tell `git` which editor it should use for your commits.

* If you choose to use VS Code, you would type:

```bash
git config --global core.editor "code --wait"
```

* If you choose to use Sublime, you would type:

```bash
git config --global core.editor "subl --wait --new-window"
```

* If you choose to use Atom, you would type:

```bash
git config --global core.editor "atom --wait"
```


## SSH Setup

Check that you have an ssh key setup first.  The follow command should output the contents of your public SSH key to your terminal:

```bash
cat ~/.ssh/id_rsa.pub
```
If you are getting a file not found error, perhaps you don't yet have an SSH key setup yet. Use the following command to setup your ssh key:

```bash
ssh-keygen -t rsa
```

Use all defaults, no password, for all prompts.  This is a necessary step to allow tmate.io sessions, AWS connectivity, or password-less Github Enterprise interactivity or any future interconnectivity with secure shell sessions.


That's it! Now you're ready to begin GA's Data Science Immersive!

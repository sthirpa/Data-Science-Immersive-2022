### How to craft a Markdown file

A type of file you may have been unfamiliar with until now is the MarkDown file. Markdown files are specified with the .md suffix. This suffix instructs programs to render them in the MarkDown format. When you first start a new markdown file, in many regards it will be similar to a normal text file.

You can easily create a new markdown file in your terminal with the following command:

`$ touch filename.md`

This will create a file called `filename.md`. You may want to name your markdown file something related to the contents of the file, such as `executive_summary.md` or `technical_report.md`.

In order to edit the file, you can open it from command line using `open` or the editor of your choice. Below is an example using atom.

`$ atom filename.md`

Now that the file is open, you can start editing it. Most of the time, it will just be text. There are a few things you can do however that are special to markdown.

`#` are for section headers. 1 # will create a large header, and multiple ## will create slightly smaller headers.

#### This is what #### looks like

You can also create tables in markdown. Here is an example of a well formatted table.

|             	| column 1   	| column 2 	|
|-------------	|-----------	|--------	|
| Row 1      	| 1         	| 1         |
| Row 2      	| 1         	| 1         |
| Row 3      	| 1         	| 1         |

The above table was generated with the following text:
```
|            	| column 1      | column 2  |
|------------   |-----------    |--------   |
| Row 1      	| 1         	| 1         |
| Row 2      	| 1         	| 1         |
| Row 3      	| 1         	| 1         |
```

You can try to line up all the | but you dont have to, it will still render.

Finally, you may want to include images in your markdown file, especially if it is your executive summary. Here's how:

Save the image as a .png to a directory in your project named visuals `$ mkdir visuals`. Git add, commit, and push the visuals to github. Now add a link to your visual from the markdown file:

`![image_title](https://github_url_to_image.png)`

If you've done it correctly, your image should display when the markdown file renders on GitHub.

### How to replace the README.md

You'll notice that part of the requirements for your project to be complete are that you replace the instructions that come with the project with either your own introduction to the problem statment or your entire executive summary. You may be wondering how to do this. You can either remove `rm README.md` and start fresh, or if you want to keep the instructions around for future reference, you can `cp README.md instructions.md`. If you choose the second, you should also take the following two steps:

`$ touch .gitignore`

`$ open .gitignore`

Paste `instructions.md` in the open .gitignore, save and close. Now when you commit your work the instructions wont be committed with it. Now you are ready to get started writing your executive summary! 

# Detailed AWS Steps

## TURN ON BILLING ALERTS
https://console.aws.amazon.com/billing/home?#/preferences
- Check the box "Receive Free Tier Usage Alerts"

## LOG IN TO AWS DASHBOARD
https://us-west-2.console.aws.amazon.com/console/home?region=us-west-2

Services (upper right)
-> EC2

Click on Instances (left bar)
-> 'Launch instances' button

1. Select "Ubuntu Server" - Make sure it says Free tier!

2. Select "t2.micro" - Make sure it says Free tier!

3. Steps 3-5 - Do not make modifications.

4. Step 6: Configure Security Group
- Change SSH Source to "Anywhere"
- Add a Rule. 
	- Select "Custom TCP Rule"
	- Change Port Range to 8888
	- Change Source to "Anywhere"

5. Launch our instance
- Select "Create a new key pair"
- Give it a name "my-first-key"
- Click "Download Key Pair"
- Drag the file to your Desktop
- Click "Launch Instances"

6. Click on "View Instances" on the next screen.

7. Click on your instance then get the "Public IPv4 DNS", e.g.:
`ec2-34-216-192-165.us-west-2.compute.amazonaws.com`
- Below, this is referred to as `<COMPUTER NAME>`

## LOG IN TO THE INSTANCE
1. Open Terminal
- `cd Desktop`
- `chmod 600 my-first-key.pem`
- `ssh -i my-first-key.pem ubuntu@<COMPUTER NAME>`

2. Update Software to latest versions
- `sudo apt-get update`
- `sudo apt-get upgrade`

3. View Log files
- `more /var/log/auth.log`  # Attempts to login to your system
- `more /var/log/syslog`    # Why your computer crashed
- `tail /var/log/syslog`    # Only look at the latest lines

4. Install Anaconda
- `wget http://repo.continuum.io/archive/Anaconda3-2020.07-Linux-x86_64.sh`
- `bash Anaconda3-2020.07-Linux-x86_64.sh`
  - Press Enter. Review the license agreement.
  - Type `yes` to accept the license terms.
  - Press Enter to install in the default directory.
  - Type `yes` to have Anaconda init a new environment.

5. Run the 
`source .bashrc`

6. Run: `jupyter notebook --ip=*`
- In your browser, visit: `<COMPUTER NAME>:8888`
- The token will be in your console window!


## Copying Files (also can use Filezilla or Jupyter upload/download)
- To copy a file to the remote computer:

`scp -i ~/Desktop/my-first-key.pem <FILE TO COPY> ubuntu@<COMPUTER NAME>:~`

- To copy a file from the remote computer (to the current directory, i.e. the final .):

`scp -i ~/Desktop/my-first-key.pem  ubuntu@<COMPUTER NAME>:~/Untitled.ipynb .`

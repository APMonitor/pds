import pandas as pd
import smtplib
from email.mime.text import MIMEText
from getpass import getpass

ask = False
if ask:
    From = input("Enter email address of the sender: ")
    username = input("Enter email user name: ")
    smtp_server = input("Enter SMTP server address: ")
    password = getpass("Password for "+username+" at "+smtp_server+": ")
else:
    From ='Instructor <instructor@example.com>'
    username ='my_username'
    smtp_server ='mail.et.byu.edu'
    password = '1234' # not good practice to put password in the code

url = 'http://apmonitor.com/pds/uploads/Main/students.txt'
students = pd.read_csv(url)

def sendEmail(Subject, bodyText, To, pw):
    msg = MIMEText(bodyText)
    msg['Subject'] = Subject
    msg['From']    = From
    msg['To']      = To

    server = smtplib.SMTP(smtp_server)
    server.starttls()
    server.login(username, password)
    server.send_message(msg)
    server.quit()

    return 'Sent to ' + To

Message = '''We missed you in class today. I hope you are doing well.

Today we worked on the project for facial recognition.

Best regards,

John Hedengren
Brigham Young University'''

for i in range(len(students)):
    bdTxt = students.First[i] + ',\n\n' + Message

    To = students.Email[i]
    print(To)
    Subject = "Hi " + students.First[i] + ", we missed you today"
    sendEmail(Subject,bdTxt,To,password)
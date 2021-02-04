import yagmail
from datetime import date
import os
import time

current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)

output_dir = os.path.join(root_dir, 'output')
notebook_path = os.path.join(output_dir, "Nuove Case.pdf")

mail_list = [
    'lorenza.Grappeja.lg@gmail.com',
    'davide.stenner@gmail.com'
]
print('Sending email...\n')
today = date.today()

yag = yagmail.SMTP("stennerpy@gmail.com")

CONTENTS = ['Case nuove in zona Barona/Famagosta.', yagmail.inline(notebook_path)]

for email in mail_list:
    yag.send(email, "Nuove Case", CONTENTS)
    time.sleep(3)
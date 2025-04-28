In this project, for the purpose of fraud transaction detection, the time elapsed between two consecutive transactions is taken into account. 
If the time elapsed between the transactions is not within the specified limit, the users are required to initially enter their details in a Gateway Interface. 
This indirectly serves the purpose of cross-verification with the genuine cardholder details.
Proceeding further, a One-Time Password or OTP is delivered to the authorized user’s Telegram Account. This feature was developed using the Telegram Bot. 
As an additional and integral procedure of genuine user verification, a face recognition system is implemented next, and there the authorized user’s face is already recorded and stored in the database.
This particular feature is implemented using the OpenCV model.
If the user’s transaction is not within the specified safe time limit, it is considered to be a fraudulent transaction in the very first place. 
If the user’s transaction is within the specified time limit, then the user is required to initially enter their details in a gateway,  enter a genuine OTP received on their Telegram Account 
and finally bypass the face authentication system.
If the user fails to either enter the genuine OTP or is unable to bypass the face authentication system or fails in both these steps,
in all these three circumstances, the user cannot complete the transaction and the transaction is classified to be a fraudulent transaction.
For the purpose of recording and storing the genuine user’s facial data, files namely training.py, face_recognition.py and face_dataset.py files are employed. 
The Naive Bayes Algorithm is used to train the model.
The project can be executed by executing the app.py file. 
 
 

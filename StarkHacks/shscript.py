import serial
import time

ser = serial.Serial('COMX', 9600, timeout=2) #check device manager for X value
time.sleep(2) #Arduino reset time

ser.write(b"ON\n")
reply = ser.readline()
print(reply.decode().strip()) #prints response (should be ACK)

time.sleep(3)

ser.write(b"OFF\n")
reply = ser.readline()
print(reply.decode().strip())


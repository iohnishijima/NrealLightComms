from device_connection import NrealLight
from other_methods import methods
import time

def main():
    try:
        flag = False
        nreal_light = NrealLight()
        print("Device initialized and connected.")
        #Get serial number 
        serialNumber = methods.parse_response(nreal_light.send_command('3', 'C'))
        print(f"serial number is: {serialNumber}")
        #Get firmware version
        firmware = methods.parse_response(nreal_light.send_command('3', '5'))
        print(f"firmware is: {firmware}")
        #Get Display mode
        displayMode = methods.parse_response(nreal_light.send_command('3', '3'))
        print(f"display mode is: {displayMode}")
        #Keep RGB camera open
        RGBCamera = methods.parse_response(nreal_light.send_command('1', 'h', '1'))
        print(f"RGB camera is: {RGBCamera}")
        #Check SDK is working
        SDKWorks = methods.parse_response(nreal_light.send_command('@', '3', '1'))
        print(f"SDK works is: {SDKWorks}")
        #Set Display mode to 3D
        setDisplayMode = methods.parse_response(nreal_light.send_command('1', '3', '3'))
        print(f"desplay mode set: {setDisplayMode}")
        #parse heartbeat every less than 1 sec
        heartbeat = methods.parse_response(nreal_light.send_command('@', 'K'))
        print(f"heartbeat  is: {heartbeat}")
        
        lastTime = time.time()
        print(lastTime)
        while True:
            if not flag:
                print("Loop running... ")
                flag = True
            currentTime = time.time()
            if 0.9 <= currentTime - lastTime < 1.0:
                print(f"already past {time.time() - lastTime}s")
                print("hearbeat")
                methods.parse_response(nreal_light.send_command('@', 'K'))
                lastTime = currentTime
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
import p11_car
import p12_tv

print('==================')

print(" do.py와 module 이름은", __name__)  

print('==================')

p11_car.drive()
p12_tv.watch()


'''
운전하다
 car.py와 module 이름은 p11_car
시청하다
 tv.py와 module 이름은 p12_tv
==================
 do.py와 module 이름은 __main__  <-  ctrl +F5 누른 파일이 항상 main 
==================
운전하다
시청하다

'''
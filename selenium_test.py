from selenium import webdriver
from selenium.webdriver.common.by import By
import time

browser = webdriver.Chrome()
browser.set_window_size(1920, 1080)
browser.get('https://mail.163.com/')

iframe = browser.find_element(By.XPATH, '//div[@id="loginDiv"]/iframe')
print(iframe)
#表单切换
browser.switch_to.frame(iframe)
#定位账号输入框并输入
browser.find_element(By.NAME, 'email').send_keys("tonkit")
#定位密码输入框并输入
browser.find_element(By.NAME, 'password').send_keys("Test_tonkit1")
time.sleep(1)
#点击登录按钮
browser.find_element(By.ID, 'dologin').click()
time.sleep(1)

browser.find_element(By.ID, '_mail_component_149_149').click()
time.sleep(1)
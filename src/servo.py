class Servo:
    def __init__(self,min_ang,max_ang,current_ang):
        self.min_ang=min_ang
        self.current_ang=current_ang
        self.max_ang=max_ang
        
    def move_to(self,target_ang,step_ang):
        if target_ang < self.min_ang or target_ang > self.max_ang:
            return 'Servo cannot move to the target angle'
        while self.current_ang != target_ang:
            if self.current_ang < target_ang:
                     if target_ang-self.current_ang<step_ang:
                          step_ang=target_ang-self.current_ang
                     self.current_ang+=step_ang
            if self.current_ang > target_ang:
                     if self.current_ang-target_ang<step_ang:
                          step_ang=self.current_ang-target_ang
                     self.current_ang-=step_ang
        print('current angle=',self.current_ang)

             #min max current       
servo1=Servo(20,90,45)
servo1.move_to(80,5)     #target angle step angle
servo2=Servo(20,90,21)
servo2.move_to(20,5)




        
        
    



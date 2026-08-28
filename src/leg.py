from servo import Servo
import math 
class Leg:
    def __init__(self,coxa_angle,femur_angle,tibia_angle,coxa,femur,tibia):
        self.coxa_angle=coxa_angle
        self.femur_angle=femur_angle
        self.tibia_angle=tibia_angle
        self.coxa=coxa
        self.femur=femur
        self.tibia=tibia
    def forward_kinematics(self): # ANGLE-->(X,Y,Z)
        xf=self.femur*(math.cos((self.femur_angle*math.pi)/180)) # forward contri due to femur
        zf=self.femur*(math.sin((self.femur_angle*math.pi)/180)) #vertical contri due to femur
        xt=self.tibia*(math.cos((self.tibia_angle*math.pi)/180)) # forward contri due to tibia
        zt=self.tibia*(math.sin((self.tibia_angle*math.pi)/180))  # vertical contri due to tibia

        total_horizontal_reach=xf+xt
        total_vertical_reach=zt+zf
        radial_reach_of_coxa=self.coxa+total_horizontal_reach

        x=radial_reach_of_coxa*(math.cos((self.coxa_angle*math.pi)/180))
        y=radial_reach_of_coxa*(math.sin((self.coxa_angle*math.pi)/180))

        return '({},{},{})'.format(x,y,total_vertical_reach)

    def inverse_kinematics(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z

        coxa_angle = math.degrees(math.atan2(self.y, self.x))

        coxa_joint_to_foot_horizontal = math.sqrt(self.x**2 + self.y**2) #10
        femur_joint_to_foot_horizontal=coxa_joint_to_foot_horizontal-self.coxa #7
        target_line= math.sqrt(femur_joint_to_foot_horizontal**2 + self.z**2)
        if target_line>self.femur+self.tibia or target_line<math.fabs(self.tibia-self.femur):
            return 'Target position is unreachable'

        knee_angle=math.degrees(math.acos((self.femur**2+self.tibia**2-target_line**2)/(2*self.tibia*self.femur)))

        femur_joint_triangle_angle=math.degrees(math.acos((self.femur**2+target_line**2-self.tibia**2)/(2*self.femur*target_line))) #beta#angle between femur and tibia from femur-tibia-targetline triangle
        target_line_horizontal_line_angle=math.degrees(math.atan2(self.z,femur_joint_to_foot_horizontal)) #alpha#angle bw target line(8.06) and horizontal line

        femur_angle=femur_joint_triangle_angle+target_line_horizontal_line_angle

        tibia_angle = (femur_angle+180 + knee_angle) % 360 #39.3+180 <--- reverses the femur joint to tibia joint vector and then adding the knee angle gives us the angle made by tibia with the horizontal

        return (coxa_angle,femur_angle,tibia_angle)

coxa=Servo(40,145,90)
femur=Servo(30,90,40)
tibia=Servo(15,90,60)

leg1=Leg(coxa,femur,tibia,3,5.6,8)

    



        







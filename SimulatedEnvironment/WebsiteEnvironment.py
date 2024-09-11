from VoEyeur import *

class User:

    def __init__(self,username):
        self.username=username

    def wait_user_action(self):
        decision=input("Where would you like to go?")
        return decision


class WebsiteEnvironment:
    
    def __init__(self,sections,user):
        self.sections=sections
        self.voeyeur=VoEyeur(sections)
        self.user=user


    def start_environment(self):
        state={}
        state["suggest_time"]=self.voeyeur.study_time # When VoEyeur starts, it studies user navigation patterns
        state["prev_sug_sec"]=self.sections[0]
        state["previous_visit"]=self.sections[0]
        state["visited_section"]=self.sections[0]
        environment_existing=True
        while (environment_existing):
            action_cause=self.voeyeur.be_alive(state,discovery=True)
            if(action_cause):
                state["prev_sug_sec"]=action_cause["suggestion"]

            # User choice will be remembered in the next time step
            state["previous_visit"]=state["visited_section"]
            state["visited_section"]=self.user.wait_user_action()
            if(state["suggest_time"]>=self.voeyeur.study_time+self.voeyeur.think_time):
                state["suggest_time"]=0
                continue
            state["suggest_time"]+=1

sections=["Homepage","Services"] # Minimum two sections required
website_environment=WebsiteEnvironment(sections,User("Agent K"))

website_environment.start_environment()

    
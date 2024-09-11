import numpy as np


from pomdpuzzle.dynamics.dynamics_utils import *
from pomdpuzzle.dynamics.policy_utils import *
from pomdpuzzle.solver.approximations.quadr_value_iterator import *
from pomdpuzzle.visualization.boundaries_viewer import *


class VoEyeur:

    def __init__(self,sections,study_time=10,think_time=7):
        print("VoEyeur is opening its eye...")
        self.inflate(sections,study_time,think_time)
        print("VoEyeur is watching now!")
        self.speak("*Blink* Gnnn...I was sleepy... JOKE'S ON YOU! Me always peeping, mate.")
        self.appearance(2)


    def be_alive(self,environment_perception,discovery=True):
        action_cause={}
        print("Time to blllllink more!")
        suggest_time=environment_perception["suggest_time"]
        previous_suggested_section=environment_perception["prev_sug_sec"]
        visited_section=environment_perception["visited_section"]
        previous_visit=environment_perception["previous_visit"]

        if(visited_section not in self.sections):
            if(not discovery):
                return
            self.discover(visited_section,previous_suggested_section,previous_visit)

        # Check if it's time to suggest, or to spy
        if(suggest_time>=self.study_time):
            # VoEyeur gains knowledge. We consider a convex combination of previous knowledge and newly formed knowledge
            if(suggest_time==self.study_time):
                self.old_knowledge=self.user_dynamics
                self.user_dynamics=self.initialize_knowledge()
            if(suggest_time<self.study_time+self.think_time):
                # VoEyeur spies ===> Update Dynamics
                self.user_dynamics=self.spy(environment_perception)
                # VoEyeur does a fake suggestion
                self.infer(previous_suggested_section,visited_section)
                action_cause["suggestion"]=self.suggest(fake_suggestion=True)
                action_cause["value"]=self.max_value_function.evaluate(self.belief_states[:self.known_belief_policies-1])
            else:
                # VoEyeur thinks ===> Value Iteration
                self.integrate_new_knowledge()
                self.max_value_function,partition_dict,_,self.optimal_policy=self.think()
            

        else:
            # VoEyeur analyzes its previous suggestion, observes where User landed and infers new beliefs
            print(self.user_dynamics.observation_state_action_prob[:,:,0])
            self.infer(previous_suggested_section,visited_section)

            # VoEyeur suggests ===> Policy Action
            action_cause["suggestion"]=self.suggest()
            action_cause["value"]=self.max_value_function.evaluate(self.belief_states[:self.known_belief_policies-1])

        return action_cause

    def infer(self,suggested_section,visited_section):
        self.belief_states=self.user_dynamics.update_beliefs(self.belief_states[:-1],suggested_section,visited_section)
        print(self.belief_states)


    def suggest(self, fake_suggestion=False):
        # There is the possibility that the existing policy does not take into account newly added sections, therefore we should consider only belief states up to previously known sections
        suggested_sections=self.optimal_policy.retrieve_action(self.belief_states[:self.known_belief_policies-1])

        if(not fake_suggestion):
            self.appearance(2)
            self.speak("*Blink* Oy mate! Why don't you give a look at "+suggested_sections+" ?")
        else:
            self.speak("*Blink* Mh. I would suggest "+suggested_sections)

        return suggested_sections
        

    def spy(self,environment_perception):
        self.appearance(1)
        self.speak("*Blink* Oh, me? Just peeping here and there, mate...")
        # We need to form the dynamics
        # That is, P(s'|s,a); P(o'|s')
        # To form P(o'|s'), we need to count how many times the user hopped from section o to section s
        # To form P(s'|s,a), we pretend VoEyeur has given a suggestion

        # We retrieve the previous suggested section as fake (pseudo) suggest
        previous_pseudo_suggest=environment_perception["prev_sug_sec"]
        
        # We get the previous visit of the user through cookies. This is both our observation and our known previous state
        previous_visit_cookie=environment_perception["previous_visit"]

        # The desired page is revealed now
        desired_page=environment_perception["visited_section"]

        prev_v_index=self.sections.index(previous_visit_cookie)
        desired_p_index=self.sections.index(desired_page)

        pseudo_sug_index=self.sections.index(previous_pseudo_suggest)

        # For the moment we approximate new probability as:
        # p(k) = 2*p(k-1)/(1+p(k-1)): When there is a visit
        # p(k) = p(k-1)/(1+p(k-1)): When there is not a visit
        
        observation_state_action_prob=self.user_dynamics.observation_state_action_prob.copy()
        state_state_action_prob=self.user_dynamics.state_state_action_prob.copy()

        for s_prime_idx,s_prime in enumerate(self.sections):
            if(s_prime_idx==desired_p_index):
                state_state_action_prob[s_prime_idx,prev_v_index,pseudo_sug_index]=2*self.user_dynamics.state_state_action_prob[s_prime_idx,prev_v_index,pseudo_sug_index]/(1+self.user_dynamics.state_state_action_prob[s_prime_idx,prev_v_index,pseudo_sug_index])
            else:
                state_state_action_prob[s_prime_idx,prev_v_index,pseudo_sug_index]=self.user_dynamics.state_state_action_prob[s_prime_idx,prev_v_index,pseudo_sug_index]/(1+self.user_dynamics.state_state_action_prob[s_prime_idx,prev_v_index,pseudo_sug_index])
        state_state_action_prob[:,prev_v_index,pseudo_sug_index]=state_state_action_prob[:,prev_v_index,pseudo_sug_index]/np.sum(state_state_action_prob[:,prev_v_index,pseudo_sug_index])
                    


        for a_idx,a in enumerate(self.sections):
            for o_idx,o in enumerate(self.sections):
                if(o_idx==prev_v_index):
                    observation_state_action_prob[o_idx,desired_p_index,a_idx]=2*self.user_dynamics.observation_state_action_prob[o_idx,desired_p_index,a_idx]/(1+self.user_dynamics.observation_state_action_prob[o_idx,desired_p_index,a_idx])
                else:
                    observation_state_action_prob[o_idx,desired_p_index,a_idx]=self.user_dynamics.observation_state_action_prob[o_idx,desired_p_index,a_idx]/(1+self.user_dynamics.observation_state_action_prob[o_idx,desired_p_index,a_idx])
            observation_state_action_prob[:,desired_p_index,a_idx]=observation_state_action_prob[:,desired_p_index,a_idx]/np.sum(observation_state_action_prob[:,desired_p_index,a_idx])


        print(observation_state_action_prob[:,:,0])
        return Dynamics(self.user_dynamics.reward_state_action_prob,state_state_action_prob,observation_state_action_prob,self.sections,self.sections,self.sections,self.user_dynamics.REWARDS)
    
    def initialize_knowledge(self):
        rewards=[-1,1]
        # Uninformative Prior
        observation_state_action_prob=np.ones((len(self.sections),len(self.sections),len(self.sections)))*1/len(self.sections)
        state_state_action_prob=np.ones((len(self.sections),len(self.sections),len(self.sections)))*1/len(self.sections)
        reward_state_action_prob=np.ones((len(rewards),len(self.sections),len(self.sections)))*0.0001
        for r_idx in range(len(rewards)):
            for s_idx,s in enumerate(self.sections):
                for a_idx,a in enumerate(self.sections):
                    if(r_idx==0 and s_idx!=a_idx):
                        reward_state_action_prob[r_idx,s_idx,a_idx]=0.9999
                    if(r_idx==1 and s_idx==a_idx):
                         reward_state_action_prob[r_idx,s_idx,a_idx]=0.9999   
        dynamics=Dynamics(reward_state_action_prob,state_state_action_prob,observation_state_action_prob,self.sections,self.sections,self.sections,rewards)

        return dynamics

    def integrate_new_knowledge(self,prev_importance=0.3,new_importance=0.7):
        print("And now...*blink*")
        # We now merge old knowledge with new knowledge.
        # There is the possibility that new knowledge comprises more states than old knowledge, so we must adapt old knowledge
        adapted_old_st_st_a=np.ones((len(self.sections),len(self.sections),len(self.sections)))*1e-4
        adapted_old_obs_st_a=np.ones((len(self.sections),len(self.sections),len(self.sections)))*1e-4
        old_knowledge_shape=self.old_knowledge.state_state_action_prob.shape
        adapted_old_st_st_a[:old_knowledge_shape[0],:old_knowledge_shape[1],:old_knowledge_shape[2]]=self.old_knowledge.state_state_action_prob
        adapted_old_obs_st_a[:old_knowledge_shape[0],:old_knowledge_shape[1],:old_knowledge_shape[2]]=self.old_knowledge.observation_state_action_prob

        self.user_dynamics.state_state_action_prob=prev_importance*adapted_old_st_st_a+new_importance*self.user_dynamics.state_state_action_prob
        self.user_dynamics.observation_state_action_prob=prev_importance*adapted_old_obs_st_a+new_importance*self.user_dynamics.observation_state_action_prob
    
    def think(self):
        self.appearance(0)
        self.speak("I see, I see... *Blink*")
        # We need to perform value iteration
        value_iterator=QuadraticValueIterator(self.user_dynamics)
        max_value,partition_dict,act_value_dict,optimal_policy=value_iterator.value_iteration(eps=0.1,gamma=0.9,max_iter=40)
        # Update belief spaces in case you previoulsy discovered newly added sections
        self.known_belief_policies=len(self.sections)
        return max_value,partition_dict,act_value_dict,optimal_policy
    

    def inflate(self,sections,study_time,think_time):
        self.sections=sections
        self.known_belief_policies=len(sections)
        self.user_dynamics=self.initialize_knowledge()
        self.old_knowledge=None
        self.optimal_policy=None
        self.study_time=study_time
        self.think_time=think_time
        # Uninformative Initial Prior
        self.belief_states=np.ones(len(self.sections))*1/len(self.sections)
        self.max_value_function,_,_,self.optimal_policy=self.think() # VoEyeur thinks a random strategy


    def speak(self,message):
        print("VoEyeur says: "+message)


    def discover(self, new_section, previous_suggested_section, previous_visit):
        self.speak("Never seen this place before. *Blink* Where are we, mate?")
        previous_visit_idx=self.sections.index(previous_visit)

        # Add in sections
        self.sections.append(new_section)

        prev_sug_idx=self.sections.index(previous_suggested_section)
        
        old_obs_st=self.user_dynamics.observation_state_action_prob[:,:,0]
        # Expand P(o|s) with new matrix
        # Informative Prior: we know that the new section desire was affected by the previous visit
        new_col=np.ones((len(self.sections)-1))
        new_col[previous_visit_idx]+=1
        new_col/=np.sum(new_col)
        old_obs_st=np.append(old_obs_st, new_col[None].transpose() ,axis=1)

        # Expand P(o|s) with new row
        # Uninformative Prior: we don't know how the new inserted section leads to next sections desires

        new_row=np.ones((len(self.sections)))*1e-4
        old_obs_st=np.append(old_obs_st,new_row[None],axis=0)
        for c in range(len(self.sections)):
            old_obs_st[:,c]/=np.sum(old_obs_st[:,c])


        # Now we have to form P(o|s,a) by expanding the dims
        self.user_dynamics.observation_state_action_prob=np.repeat(old_obs_st[...,None],len(self.sections),axis=-1)

        old_st_st_act=self.user_dynamics.state_state_action_prob
        # Expand P(s'|s,a) with new (s,a) matrix
        # Informative Prior: we know that we applied suggestion a before in the previous state desire s, and then arrived in new section s'
        # In other cases we assume low probability

        new_matr=np.ones((len(self.sections)-1,len(self.sections)-1))*1e-4
        new_matr[previous_visit_idx,prev_sug_idx]=1
        old_st_st_act=np.append(old_st_st_act,new_matr[None],axis=0)
        old_st_st_act[:,previous_visit_idx,prev_sug_idx]/=np.sum(old_st_st_act[:,previous_visit_idx,prev_sug_idx])

        # Expand P(s'|s,a) with new (s',a) matrix
        # Uninformative Prior: Even if we know that we were before in s=new_suggestion and even if we know that previous suggestion was a generic a, we do not now where we will land

        new_matr=np.ones((len(self.sections),len(self.sections)-1))*1/len(self.sections)
        old_st_st_act=np.append(old_st_st_act,new_matr[:,None],axis=1)

        # Expand P(s'|s,a) with new (s',s) matrix
        # Uninformative Prior: we don't know what will happen if we suggest the new section. 
        new_matr=np.ones((len(self.sections),len(self.sections)))*1/len(self.sections)
        old_st_st_act=np.append(old_st_st_act,new_matr[:,:,None],axis=2)     
        self.user_dynamics.state_state_action_prob=old_st_st_act

        old_r_st_act=self.user_dynamics.reward_state_action_prob
        # Expand P(r|s,a) with new (r,a) matrix
        # Informative Prior: we know that suggesting the new section in everything but the new section gives bad reward
        new_matr=np.ones((2,len(self.sections)-1))*0.9999
        new_matr[1,:]=np.ones((len(self.sections)-1))*1e-4

        old_r_st_act=np.append(old_r_st_act,new_matr[:,None],axis=1)

        # Expand P(r|s,a) with new (r,s) matrix
        # Informative Prior: we know that being in the new section and suggesting everything but the new section gives bad reward
        new_matr=np.ones((2,len(self.sections)))*0.9999
        new_matr[1,:-1]=np.ones((len(self.sections)-1))*1e-4
        new_matr[0,-1]=1e-4
        old_r_st_act=np.append(old_r_st_act,new_matr[:,:,None],axis=2)

        self.user_dynamics.reward_state_action_prob=old_r_st_act
        self.user_dynamics.expected_rewards=self.user_dynamics.get_expected_rewards()

        # We need to expand the belief spaces. New belief about new section is unknown, we go for uninformative prior.
        self.belief_states=np.append(self.belief_states,1e-4)

        return None
    
    def appearance(self,type):
        if(type==0):
            print("You see VoEyeur in the shape of a lightbulb...?")
            return
        if(type==1):
            print("You see VoEyeur's pupil going wild!")
            return
        if(type==2):
            print("You see VoEyeur bouncing on the spot.")
            return
        

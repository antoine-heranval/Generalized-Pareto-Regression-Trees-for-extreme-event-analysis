
source("./simu_experience.R")
source("./simu_cart.R")

###CHOICE

source("./function_generalized_pareto_regression_tree_xi_in_R_plus_simu.R")
#source("./function_generalized_pareto_regression_tree_xi_in_R_plus_simu_big_data.R")

###CHOICE
#CART---------
source("./simu_data_step_w2.R")
#source("./simu_data_new_design.R")

start_experience=1
nb_experience=10
num_eval=100 #or NA
sigma_tan_h_1=10
sigma_tan_h_2=+Inf
kn=100
#or +Inf
#
# exp_CART_10_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_50_burr,"./exp_CART_10_50_burr.csv")
# write.csv2(exp_CART_50_50_burr,"./exp_CART_50_50_burr.csv")
#
# exp_CART_10_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
#
# write.csv2(exp_CART_10_100_burr,"./exp_CART_10_100_burr.csv")
# write.csv2(exp_CART_50_100_burr,"./exp_CART_50_100_burr.csv")
#
#
# exp_CART_10_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_250_burr,"./exp_CART_10_250_burr.csv")
# write.csv2(exp_CART_50_250_burr,"./exp_CART_50_250_burr.csv")

# exp_CART_10_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_500_burr,"./exp_CART_10_500_burr.csv")
# write.csv2(exp_CART_50_500_burr,"./exp_CART_50_500_burr.csv")
#
#
# exp_CART_10_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_1000_burr,"./exp_CART_10_1000_burr.csv")
# write.csv2(exp_CART_50_1000_burr,"./exp_CART_50_1000_burr.csv")

#
# exp_CART_10_2500_burr=experience_burr(n=2500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
exp_CART_50_2500_burr=experience_burr(n=kn,start_experience=start_experience,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)

# write.csv2(exp_CART_10_2500_burr,"./exp_CART_10_2500_burr.csv")
# write.csv2(exp_CART_50_2500_burr,"./exp_CART_50_2500_burr.csv")
#
# exp_CART_10_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_5000_burr,"./exp_CART_10_5000_burr.csv")
# write.csv2(exp_CART_50_5000_burr,"./exp_CART_50_5000_burr.csv")
#
# exp_CART_10_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
# exp_CART_50_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="CART",num_eval=num_eval)
#
# write.csv2(exp_CART_10_10000_burr,"./exp_CART_10_10000_burr.csv")
# write.csv2(exp_CART_50_10000_burr,"./exp_CART_50_10000_burr.csv")


save.image(file="./main_CART_burr_2.RData")

## GAM ---------------------

source("./simu_experience.R")

source("./simu_data_step_w.R")
#source("./simu_data_new_design.R")

source("./simu_QRM.R")

# start_experience=1
# nb_experience=3
# num_eval=10 #or NA
# sigma_tan_h_1=10
# sigma_tan_h_2=+Inf #or +Inf
#
# exp_GAM_10_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
# exp_GAM_50_50_burr=experience_burr(n=50,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
#
# write.csv2(exp_GAM_10_50_burr,"./exp_GAM_10_50_burr.csv")
# write.csv2(exp_GAM_50_50_burr,"./exp_GAM_50_50_burr.csv")
#
# exp_GAM_10_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
# exp_GAM_50_100_burr=experience_burr(n=100,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
#
#
# write.csv2(exp_GAM_10_100_burr,"./exp_GAM_10_100_burr.csv")
# write.csv2(exp_GAM_50_100_burr,"./exp_GAM_50_100_burr.csv")
#
# #
# exp_GAM_10_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
# exp_GAM_50_250_burr=experience_burr(n=250,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
#
# write.csv2(exp_GAM_10_250_burr,"./exp_GAM_10_250_burr.csv")
# write.csv2(exp_GAM_50_250_burr,"./exp_GAM_50_250_burr.csv")

# exp_GAM_10_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
# exp_GAM_50_500_burr=experience_burr(n=500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
#
# write.csv2(exp_GAM_10_500_burr,"./exp_GAM_10_500_burr.csv")
# write.csv2(exp_GAM_50_500_burr,"./exp_GAM_50_500_burr.csv")
#
#
# exp_GAM_10_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
# exp_GAM_50_1000_burr=experience_burr(n=1000,start_experience=start_experience,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
#
# write.csv2(exp_GAM_10_1000_burr,"./exp_GAM_10_1000_burr.csv")
# write.csv2(exp_GAM_50_1000_burr,"./exp_GAM_50_1000_burr.csv")
#
# exp_GAM_10_2500_burr=experience_burr(n=2500,start_experience=start_experience,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
experience_burr(n=kn,start_experience=start_experience,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)

# write.csv2(exp_GAM_10_2500_burr,"./exp_GAM_10_2500_burr.csv")
# write.csv2(exp_GAM_50_2500_burr,"./exp_GAM_50_2500_burr.csv")

#
# exp_GAM_10_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
# exp_GAM_50_5000_burr=experience_burr(n=5000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
#
#
# write.csv2(exp_GAM_10_5000_burr,"./exp_GAM_10_5000_burr.csv")
# write.csv2(exp_GAM_50_5000_burr,"./exp_GAM_50_5000_burr.csv")
#
# exp_GAM_10_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_1,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
# exp_GAM_50_10000_burr=experience_burr(n=10000,sigma_tan_h=sigma_tan_h_2,nb_experience=nb_experience,learning_method="GAM",num_eval=num_eval)
#
# write.csv2(exp_GAM_10_10000_burr,"./exp_GAM_10_10000_burr.csv")
# write.csv2(exp_GAM_50_10000_burr,"./exp_GAM_50_10000_burr.csv")
#



save.image(file="./main_GAM_burr.RData")


# Analyse -----------------------
perf_analysis_burr_2=function(learning_method,learning_model,num_eval,data_sample){
  
  
  x <- seq(0, 1, l = num_eval)
  l <- 1/(num_eval-1)
  
  gamma_true <-step_w1(x)
  sigma_true <- (step_w1(x)/(2^step_w1(x)-1))
  data_test=data.frame(X=x)
  
  #Prediction
  
  if(learning_method=="GPD"){
    parameters_estimated=matrix(learning_model,nrow=nrow(data_test),ncol=2,byrow=TRUE)
  }else{
    if(learning_method=="CART"){
      parameters_estimated=pred_CART_GPD(tree=learning_model,data_test=data_test)
    }else{
      if(learning_method=="Kernel"){
        parameters_tmp=learning_model
        x_kernel=seq(0,1,length.out=nrow(parameters_tmp))
        parameters_estimated=NULL
        for(j in 2:length(x_kernel)){
          n_x=sum(x>= x_kernel[j-1] & x<x_kernel[j])
          parameters_estimated=rbind(parameters_estimated,matrix(parameters_tmp[j-1,],nrow=n_x,ncol=ncol(parameters_tmp),byrow=TRUE))
        }
        parameters_estimated=rbind(parameters_estimated,parameters_tmp[nrow(parameters_tmp),])
        
      }else{
        if(learning_method=="GAM"){
          parameters_estimated=pred_GAM(fit=learning_model,data_test=data_test)
        }else{print("Learning method not allowed")
          return(NULL)}
      }
    }
  }
  
  #Analysis
  
  res_gamma=c(l*sum((parameters_estimated[,2]-gamma_true)^2),
              #median((parameters_estimated[,2]-gamma_true)^2),
              l*sum(abs(parameters_estimated[,2]-gamma_true)),
              #median(abs(parameters_estimated[,2]-gamma_true)),
              max(abs(parameters_estimated[,2]-gamma_true)))
  res_sigma=c(l*sum((parameters_estimated[,1]-sigma_true)^2),
              #median((parameters_estimated[,2]-gamma_true)^2),
              l*sum(abs(parameters_estimated[,1]-sigma_true)),
              #median(abs(parameters_estimated[,2]-gamma_true)),
              max(abs(parameters_estimated[,1]-sigma_true)))
  
  data_sample_perf=simulate_data_burr(n=kn,seed=1)
  Y <- list()
  x <- seq(0, 1, l = num_eval)
  LL_res_true_par_estimated=double(length(x))
  for(col in 1:(num_eval-1)){
    Y[[col]]<-data_sample_perf$Y[data_sample_perf$X>x[col] & data_sample_perf$X<=x[col+1]]
  }
  for(col in 1:(length(x)-1)){
    if(length(Y[[col]])>0){
      
      LL_tmp=log(dgpd(x = Y[[col]],
                      mu = 0,
                      beta = parameters_estimated[col,1],
                      xi = parameters_estimated[col,2]))
      
      if(sum(is.na(LL_tmp))>0){
        print(paste0("Warning",learning_method,"_",col))
      }
      
      
      LL_tmp[is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf)]=min(LL_tmp[!(is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf))])
      LL_res_true_par_estimated[col]=sum(LL_tmp)
      
    }else{LL_res_true_par_estimated[col]=0}
  }
  
  res_all=c(sum(LL_res_true_par_estimated,na.rm=TRUE)/nrow(data_sample_perf))
  return(list(res_sigma,res_gamma,res_all))
}



from_models_to_save_learning_model=function(){
  
  selection_process=function(ll_cv){
    #pos=20
    pos=match(TRUE,cumsum(ll_cv==min(ll_cv))==sum(ll_cv==min(ll_cv)))[1]
    #pos=pos-1+match(TRUE,ll_cv[pos:length(ll_cv)]>(min(ll_cv)+(ll_cv[1]-min(ll_cv))*0.05))
    #pos=match(TRUE,ll_cv[-1]>ll_cv[-length(ll_cv)])
    if(is.na(pos)){
      #If the test error does not increase, we choose the minimal deepth that minimize the test error
      pos=match(TRUE,ll_cv==min(ll_cv))[1]
      if(is.na(pos)){
        pos=0
      }
    }
    return(pos)
  }
  
  nb_experience=nb_experience
  start_experience=start_experience
  list_method=c("CART","GAM")
  list_n=c(kn)
  list_sigma_h=c(Inf)
  
  for(learning_method in list_method){
    for(n in list_n){
        assign("models",readRDS(file=paste0("./models_exp_",learning_method,"_design","_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds")))
        learning_model=list()
        for(i in 1:nb_experience){
          if(learning_method=="GPD"){
            learning_model[[i]]=models[[i]]
          }else{
            if(learning_method=="CART"){
              beta=models[[i]][[3]][[1]]
              Risk_beta=models[[i]][[3]][[2]]
              selected=selection_process(Risk_beta)
              alpha=beta[selected][1]
              GPRT=prune_tree(rpart_object=models[[i]][[2]],CP=alpha)
              learning_model[[i]]=GPRT
            }else{
              if(learning_method=="Kernel"){
                ll_cv=models[[i]][[3]]
                selected=selection_process(-ll_cv)+1
                learning_model[[i]]=cbind(models[[i]][[2]][[1]][,selected],models[[i]][[2]][[2]][,selected])
              }else{
                if(learning_method=="GAM"){
                  ll_cv=models[[i]][[3]]
                  selected=selection_process(-ll_cv)
                  learning_model[[i]]=models[[i]][[2]][[selected]]
                }else{print("Learning method not allowed")
                  return(NULL)}
              }
            }
          }
        }
        saveRDS(learning_model,file=paste0("./learning_model_exp_",learning_method,"_design","_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds"))
      }
    }
  }


from_models_to_save_learning_model()

from_learning_model_to_metrics_2=function(){
  nb_experience=nb_experience
  start_experience=start_experience
  list_method=c( "CART","GAM")
  list_n=c(kn)
  list_sigma_h=c(Inf)
  num_eval=100
  
  for(learning_method in list_method){
    for(n in list_n){
        assign("save_learning_model",readRDS(file=paste0("./learning_model_exp_",learning_method,"_design","_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds")))
        
        perf_sigma=matrix(nrow=length(save_learning_model),ncol=3)
        perf_gamma=matrix(nrow=length(save_learning_model),ncol=3)
        perf_all=matrix(nrow=length(save_learning_model),ncol=1)
        
        for(exp in 1:length(save_learning_model)){
          
          learning_model=save_learning_model[[exp]]
          
          perf=perf_analysis_burr_2(learning_method,learning_model,num_eval=num_eval,data_sample=NULL)
          
          perf_sigma[exp,]=perf[[1]]
          perf_gamma[exp,]=perf[[2]]
          perf_all[exp,]=perf[[3]]
          
        }
        
        write.csv2(perf_gamma,paste0("./MSE_GAMMA_2_exp_",learning_method,"_design","_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.csv"))
        write.csv2(perf_sigma,paste0("./MSE_SIGMA_2_exp_",learning_method,"_design","_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.csv"))
        
      }
    }
  }

from_learning_model_to_metrics_2()


experience_burr=function(n=1000,start_experience=1,nb_experience=10,learning_method="CART",num_eval=100){

  # perf_sigma=matrix(nrow=nb_experience,ncol=5)
  # perf_gamma=matrix(nrow=nb_experience,ncol=5)
  # perf_all=matrix(nrow=nb_experience,ncol=2)

  save_models=list()
  i=0
  for(seed in start_experience:(start_experience+nb_experience-1)){
    #Simulation
    i=i+1
    data_simu=simulate_data_burr(n=n,seed=seed)

    #Learning

    if(learning_method=="GPD"){
      models=gpd(data_simu$Y,threshold=0,method="ml")$par.ests[c(2,1)]
    }else{
      if(learning_method=="CART"){
      models=learning_CART_GPD(data_sample=data_simu)
      }else{
        if(learning_method=="Kernel"){
          models=learning_kernel(data_sample=data_simu,num_eval=num_eval)
        }else{
          if(learning_method=="GAM"){
            models=learning_GAM(data_sample=data_simu,num_eval=num_eval)

          }else{print("Learning method not allowed")
                return(NULL)}
      }
    }
  }

  save_models[[i]]=models
    #Performence analysis

    # perf=perf_analysis_burr(sigma_tan_h,learning_method,learning_model,num_eval=num_eval,data_sample=data_simu)
    # perf_sigma[seed,]=perf[[1]]
    # perf_gamma[seed,]=perf[[2]]
    # perf_all[seed,]=perf[[3]]

    if(i%%100==0){
      saveRDS(save_models,file=paste0("./models_exp_",learning_method,"_design","_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds"))
    }

  }

  saveRDS(save_models,file=paste0("./models_exp_",learning_method,"_design","_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds"))
  # perf_sigma=matrix(nrow=length(save_learning_model),ncol=5)
  # perf_gamma=matrix(nrow=length(save_learning_model),ncol=5)
  # perf_all=matrix(nrow=length(save_learning_model),ncol=2)
  #
  # for(exp in 1:length(save_learning_model)){
  #
  #   learning_model=save_learning_model[[exp]]
  #
  #   perf=perf_analysis_burr(sigma_tan_h,learning_method,learning_model,num_eval=num_eval,data_sample=NULL)
  #
  #   perf_sigma[exp,]=perf[[1]]
  #   perf_gamma[exp,]=perf[[2]]
  #   perf_all[exp,]=perf[[3]]
  #
  # }
  #
  # write.csv2(perf_gamma,paste0("./MSE_exp_",method,"_",sigma_h,"_",n,"_burr.csv"))
  # write.csv2(perf_all,paste0("./LL_exp_",method,"_",sigma_h,"_",n,"_burr.csv"))
  # write.csv2(list(perf_sigma,perf_gamma,perf_all),paste0("./exp_",method,"_",sigma_h,"_",n,"_burr.csv"))

  return(NULL)

}

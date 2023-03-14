source("./function_generalized_pareto_regression_tree_xi_in_R_plus_simu.R")
source("./simu_kernel_0.R")
source("./simu_cart.R")
source("./simu_QRM.R")
source("./simu_data.R")

from_models_to_save_learning_model=function(){

  selection_process=function(ll_cv){
    #pos=20
    pos=match(TRUE,cumsum(ll_cv==min(ll_cv))==sum(ll_cv==min(ll_cv)))[1]
    #pos=pos-1+match(TRUE,ll_cv[pos:length(ll_cv)]>(min(ll_cv)+(ll_cv[1]-min(ll_cv))*0.05))
    #pos=match(TRUE,ll_cv[-1]>ll_cv[-length(ll_cv)])
    if(is.na(pos)){
      #If the test error does not increase, we choose the minimal deepth that minimize the test error
      #pos=match(TRUE,ll_cv==min(ll_cv))[1]
      if(is.na(pos)){
        pos=0
      }
    }
    return(pos)
  }

    nb_experience=2
    start_experience=1
    list_method=c("CART")
    list_n=c(100)
    list_sigma_h=c(Inf)

    for(learning_method in list_method){
      for(n in list_n){
        for(sigma_tan_h in list_sigma_h){
          assign("models",readRDS(file=paste0("./models_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds")))
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
                    selected=selection_process(-ll_cv)+1
                    learning_model[[i]]=models[[i]][[2]][[selected]]
                  }else{print("Learning method not allowed")
                  return(NULL)}
                }
              }
            }
          }
          saveRDS(learning_model,file=paste0("./learning_model_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds"))
        }
      }
    }
  }




understanding_cart=function(){

    nb_experience=1000
    start_experience=2501
    list_method=c("CART")
    list_n=c(2500)
    list_sigma_h=c(Inf)
    num_eval=100

    for(learning_method in list_method){
      for(n in list_n){
        for(sigma_tan_h in list_sigma_h){
          assign("save_learning_model",readRDS(file=paste0("./learning_model_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds")))

          splits=c()

          for(exp in 1:length(save_learning_model)){
            learning_model=save_learning_model[[exp]]
            splits=c(splits,sum(learning_model$frame$var=="<leaf>")-1)
          }
          return(splits)
        }
      }
    }
  }



from_learning_model_to_metrics_2=function(){
  nb_experience=2
  start_experience=1
  list_method=c("CART")
  list_n=c(100)
  list_sigma_h=c(Inf)
  num_eval=10

  for(learning_method in list_method){
    for(n in list_n){
      for(sigma_tan_h in list_sigma_h){
        assign("save_learning_model",readRDS(file=paste0("./learning_model_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds")))

        perf_sigma=matrix(nrow=length(save_learning_model),ncol=3)
        perf_gamma=matrix(nrow=length(save_learning_model),ncol=3)
        perf_all=matrix(nrow=length(save_learning_model),ncol=1)

        for(exp in 1:length(save_learning_model)){

          learning_model=save_learning_model[[exp]]

          perf=perf_analysis_burr_2(sigma_tan_h,learning_method,learning_model,num_eval=num_eval,data_sample=NULL)

          perf_sigma[exp,]=perf[[1]]
          perf_gamma[exp,]=perf[[2]]
          perf_all[exp,]=perf[[3]]

        }

        write.csv2(perf_gamma,paste0("./MSE_2_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.csv"))
        write.csv2(perf_all,paste0("./LL_2_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.csv"))

      }
    }
  }
}

from_learning_model_to_metrics=function(){

  nb_experience=1000
  start_experience=2501
  list_method=c("CART")
  list_n=c(2500)
  list_sigma_h=c(Inf)
  num_eval=100

  for(learning_method in list_method){
    for(n in list_n){
      for(sigma_tan_h in list_sigma_h){
        assign("save_learning_model",readRDS(file=paste0("./learning_model_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.rds")))

        perf_sigma=matrix(nrow=length(save_learning_model),ncol=3)
        perf_gamma=matrix(nrow=length(save_learning_model),ncol=3)
        perf_all=matrix(nrow=length(save_learning_model),ncol=1)

        for(exp in 1:length(save_learning_model)){

          learning_model=save_learning_model[[exp]]

          perf=perf_analysis_burr(sigma_tan_h,learning_method,learning_model,num_eval=num_eval,data_sample=NULL)

          perf_sigma[exp,]=perf[[1]]
          perf_gamma[exp,]=perf[[2]]
          perf_all[exp,]=perf[[3]]

        }

        write.csv2(perf_gamma,paste0("./MSE_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.csv"))
        write.csv2(perf_all,paste0("./LL_exp_",learning_method,"_design",sigma_tan_h,"_n_data",n,"_N_exp_",nb_experience,"_start_exp_",start_experience,"_burr.csv"))

      }
    }
  }
}


perf_analysis_burr_2=function(sigma_tan_h,learning_method,learning_model,num_eval,data_sample){


  x <- seq(0, 1, l = num_eval)
  l <- 1/(num_eval-1)

  if(sigma_tan_h<+Inf){
    gamma_true = 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75))
  }else{
    gamma_true = 1 + 0.25*step_simulate_data(x-0.25) + 0.25*step_simulate_data(x-0.75)
  }

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

    data_sample_perf=simulate_data_burr(n=100000,seed=1,sigma_tan_h=sigma_tan_h)
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
  return(list(res_gamma,res_gamma,res_all))
}


perf_analysis_burr=function(sigma_tan_h,learning_method,learning_model,num_eval,data_sample){


  x <- seq(0, 1, l = num_eval)
  l <- 1/(num_eval-1)

  if(sigma_tan_h<+Inf){
    gamma_true = 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75))
  }else{
    gamma_true = 1 + 0.25*step_simulate_data(x-0.25) + 0.25*step_simulate_data(x-0.75)
  }

  data_test=data.frame(X=x)

  #Prediction

  if(learning_method=="GPD"){
    parameters_estimated=matrix(learning_model,nrow=nrow(data_test),ncol=2,byrow=TRUE)
  }else{
    if(learning_method=="CART"){
      parameters_estimated=pred_CART_GPD(tree=learning_model,data_test=data_test)
    }else{
      if(learning_method=="Kernel"){
        parameters_estimated=learning_model
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

    data_sample_perf=simulate_data_burr(n=100000,seed=1,sigma_tan_h=sigma_tan_h)
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
        LL_tmp[is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf)]=min(LL_tmp[!(is.na(LL_tmp) | LL_tmp %in% c(-Inf,Inf))])

        LL_res_true_par_estimated[col]=sum(LL_tmp)

    }else{LL_res_true_par_estimated[col]=NA}

    }

  if(sum(is.na(LL_res_true_par_estimated))>0){
    print("Warning")
  }
  res_all=c(sum(LL_res_true_par_estimated,na.rm=TRUE)/nrow(data_sample_perf))
  return(list(res_gamma,res_gamma,res_all))
}

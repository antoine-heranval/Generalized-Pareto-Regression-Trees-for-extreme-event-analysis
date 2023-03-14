library("QRM")
#
# source("./simu_data.R")
#
# eps=0.005
#
#
# fit = gamGPDfit(data_sample, threshold=0, datvar="Y", xiFrhs=as.formula("~  1 + s(X,k=5,fx=TRUE)"),nuFrhs=as.formula("~ 1 + s(X,k=5,fx=TRUE)"),
#                 eps.xi=eps, eps.nu=eps,niter = 10,progress = FALSE)
#
# fit_gamma_gam=predict(object=fit$xiObj, newdata=data_test, alpha=0.05)
# fit_sigma_gam=exp(predict(object=fit$nuObj, newdata=data_test, alpha=0.05))/(1+fit_gamma_gam)
#
#
# par(mfrow=c(1,2))
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Sigma")
# lines(x,fit_sigma_gam,col="blue")
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Gamma")
# lines(x,fit_gamma_gam,col="blue")
#



learning_GAM=function(data_sample,num_eval){

  eps=0.005
  k_seq=0:10
  k_seq_check=rep(FALSE,length(k_seq))
  fit=list()
  j=1
  
  fit[[1]]=gamGPDfit(data_sample, threshold=0, datvar="Y", xiFrhs=as.formula("~  1 "),nuFrhs=as.formula("~  1 "),
                  eps.xi=eps, eps.nu=eps,niter = 25,progress = FALSE)

  for(i in 1:(length(k_seq)+1)){
      if(i==1){
        Formula=as.formula(paste0("~  1"))
      }else{
      Formula=as.formula(paste0("~  1 + s(X,k=",k_seq[i-1],",fx=TRUE)"))
    }

      fit_tmp = gamGPDfit(data_sample, threshold=0, datvar="Y", xiFrhs=Formula,nuFrhs=Formula,
                      eps.xi=eps, eps.nu=eps,niter = 25,progress = FALSE)

      if(length(fit_tmp)==14){

        x <- seq(0, 1, l = num_eval)
        data_test=data.frame(X=x)
        parameters_estimated=pred_GAM(fit=fit_tmp,data_test=data_test)

        if(max(parameters_estimated[,2])<10 & min(parameters_estimated[,2])>-1){

          k_seq_check[i]=TRUE
          fit[[j]] = fit_tmp
          j=j+1

        }
      }
  }

# Evaluation

if(sum(k_seq_check)>0){

  folds=3
  ll_cv=matrix(nrow=folds,ncol=sum(k_seq_check))
  set.seed(1)
  cv=sample(x = 1:folds,size = nrow(data_sample),replace = TRUE)


  for(i in 1:sum(k_seq_check)){

    if(match(i,cumsum(k_seq_check))==1){
        Formula=as.formula(paste0("~  1"))
    }else{
      Formula=as.formula(paste0("~  1 + s(X,k=",k_seq[match(i,cumsum(k_seq_check))],",fx=TRUE)"))
    }


        for(j in 1:folds){

          fit_cv = gamGPDfit(data_sample[cv!=j,], threshold=0, datvar="Y", xiFrhs=Formula,nuFrhs=Formula,
                          eps.xi=eps, eps.nu=eps,niter = 25,progress = FALSE)

          if(length(fit_cv)==14){
            pred_cv=pred_GAM(fit=fit_cv,data_test=data_sample[cv==j,])



          LL=log(dgpd(x = data_sample$Y[cv==j] ,
                                mu = 0,
                                beta = pred_cv[,1],
                                xi = pred_cv[,2]))
          #LL[is.na(LL)]=min(LL,na.rm=TRUE)
          ll_cv[j,i]=sum(LL)

            # ll_cv[j,i]=sum(log(dgpd(x = data_sample$Y[cv==j] ,
            #                         mu = 0,
            #                         beta = pred_cv[,1],
            #                         xi = pred_cv[,2])))
          }else{
            ll_cv[j,i]=NA
          }


    }
  }


    ll_cv[is.na(ll_cv)]=min(ll_cv,na.rm=TRUE)-abs(min(ll_cv,na.rm=TRUE))
    ll_cv=colSums(ll_cv,na.rm=TRUE)


    if(length(which.max(ll_cv))==0){
      print("Cross validation do not reach optimal GAM fit : classic non consitional GPD adjustment applied.")

      res_fit=gamGPDfit(data_sample, threshold=0, datvar="Y", xiFrhs=as.formula("~  1 "),nuFrhs=as.formula("~  1 "),
                      eps.xi=eps, eps.nu=eps,niter = 25,progress = FALSE)
    }else{

        res_fit=fit[[which.max(ll_cv)]]
    }

    res=list(res_fit,fit,ll_cv)

}else{

  print("No GAM fit reached : classic non consitional GPD adjustment applied.")

  res_fit=gamGPDfit(data_sample, threshold=0, datvar="Y", xiFrhs=as.formula("~  1 "),nuFrhs=as.formula("~  1 "),
                  eps.xi=eps, eps.nu=eps,niter = 25,progress = FALSE)

  res=list(res_fit,fit,NULL,NULL)

}

  return(res)

}




pred_GAM=function(fit,data_test){

  fit_gamma_gam=predict(object=fit$xiObj, newdata=data_test, alpha=0.05)
  fit_sigma_gam=exp(predict(object=fit$nuObj, newdata=data_test, alpha=0.05))/(1+fit_gamma_gam)

  return(cbind(fit_sigma_gam, fit_gamma_gam))
}






#
#
#
#
# #With bootstrap
#
# fit_B = gamGPDboot(data_sample, B=100, threshold=0, datvar="Y", xiFrhs=as.formula("~ 1 + X"),nuFrhs=as.formula("~ 1 + X"),
#                    eps.xi=eps, eps.nu=eps,niter = 100,progress = FALSE)
#
#
# fit_gamma_kernel=predict(object=fit_B$bfit100$xiObj, newdata=data_test, alpha=0.05)
# fit_sigma_kernel=exp(predict(object=fit_B$bfit100$nuObj, newdata=data_test, alpha=0.05))/(1+fit_gamma_kernel)
# par(mfrow=c(1,2))
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Sigma")
# lines(x,fit_sigma_kernel,col="blue")
#
# plot(x, 1 + 0.25*tanh(sigma_tan_h*(x-0.25)) + 0.25*tanh(sigma_tan_h*(x-0.75)),type='l',col="red",xlab="x",ylab="Gamma")
# lines(x,fit_gamma_kernel,col="blue")

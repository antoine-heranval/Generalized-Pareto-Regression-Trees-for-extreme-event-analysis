# Generalized Pareto Regression Trees -------------------------------------

library(rpart)
library(testit)
library(evir)
library(treeClust)

#Functions for growing

#Initialisation function

#'@param y the response value as found in the formula. Note that rpart will normally have removed any observations with a missing response.
#'@param offset the offset term, if any, found on the right hand side of the formula
#'@param parms the vector or list (if any) supplied by the user as a parms argument to the call.
#'@param wt the weight vector from the call, if any
#'@return y the value of the response, possibly updated
#'@return numresp the length of the prediction vector for each node (here 2 for the estimates of the shape and the scale)
#'@return numy the number of columns of y
itempgprt <- function(y, offset, parms, wt) {
  if (is.matrix(y) && ncol(y) > 1)
    stop("Matrix response not allowed")
  if (!missing(parms) && length(parms) > 0)
    warning("parameter argument ignored")
  if (length(offset)) y <- y - offset
  sfun <- function(yval, dev, wt, ylevel, digits ) {
    paste(" Shape=", format(signif(yval, digits)),
          ", Deviance=" , format(signif(dev/wt, digits)),
          sep = '')
  }
  environment(sfun) <- .GlobalEnv
  list(y = c(y), parms = NULL, numresp = 2, numy = 1, summary = sfun)
}

#Evaluation function

#'@param y the response value as found in the formula. Note that rpart will normally have removed any observations with a missing response.
#'@param wt the weight vector from the call, if any
#'@param parms the vector or list (if any) supplied by the user as a parms argument to the call.
#'@return sh the estimate of the shape parameter fitted in the node
#'@return sc the estimate of the scale parameter fitted in the node
#'@return rss the loss of the node that is the negative log likelihood plus 10 times the number of observations in the node
#'The add of ten times the number of observation is for computational purposes in order to satisfy the two following conditions in the same time
#'1) to have a non negative loss
#'2) to be able to compute the performance of a split ie the comparison of the sum of losses of children nodes to the loss of the parent node
#'The factor 10 may have to be adapted depending on the data
#'(the more the value of the rss is closed to 0 the more purely the node is)
etempgprt <- function(y, wt, parms) {

  leval=FALSE
  if(!has_warning(try(evir::gpd(y, threshold = 0, method = "ml"),silent = TRUE))){
    if(!has_error(try(evir::gpd(y, threshold = 0, method = "ml"),silent = TRUE),silent = TRUE)){
      leval=TRUE
    }
  }
  if(leval){

    fit=try(evir::gpd(y, threshold = 0, method = "ml"),silent = TRUE)
    #Shape parameter
    sh=fit$par.ests[1]
    #Scale parameter
    sc=fit$par.ests[2]
    #Negative log-likelihood plus ten times the number of observation
    rss=positive_nll(fit$nllh.final)#+10*length(y)

    if(rss<0){
      #Warning if the rss is negative
      print("Warning: rss negative")
    }

  }else{
    #If the fit of the GPD does not work, we penalize the evaluation by allowing a rss equal to 100000
    sh=NA
    sc=NA
    rss=100000
    #rss=naive_ll_gpd_fit(y)
  }

  #print(paste(Sys.time(),": Computing..."))
  list(label = c(sh,sc), deviance = rss)
}

#Split function

#'@param y the response value as found in the formula. Note that rpart will normally have removed any observations with a missing response.
#'@param wt the weight vector from the call, if any
#'@param parms the vector or list (if any) supplied by the user as a parms argument to the call.
#'@param continuous if TRUE the x variable should be treated as continuous
#'@return goodness the utility of the split, where larger numbers are better. A value of 0 signifies that no worthwhile split could be found (for instance if y were a constant). The ith value of goodness compares a split of observations 1 to i versus i + 1 to n.
#'(the more the value of the goodness is closed to 0 the less purely the node is)
#'The add of ten times the number of observation is for computational purposes in order to satisfy the two following conditions in the same time
#'1) to have a non negative loss
#'2) to be able to compute the performance of a split ie the comparison of the sum of losses of children nodes to the loss of the parent node
#'The factor 10 may have to be adapted depending on the data
#'@return direction a vector of the same length with values of -1 and +1, where -1 suggests that values with y < cutpoint be sent to the left side of the tree, and a value of +1 that values with y < cutpoint be sent to the right. This is not critical, but sending larger values of y to the right, as is done in the code below, seems to make the final tree easier to read.sc the estimate of the scale parameter fitted in the node
stempgprt <- function(y, wt, x, parms, continuous)
{

  #Test of the fit of the fevd function, if it works well the local variable lparent takes the value TRUE, FALSE in the contrary case

  #Test if the fevd fit works well
  if(!has_warning(try(evir::gpd(y, threshold = 0, method = "ml"),silent = TRUE))){
    if(!has_error(try(evir::gpd(y, threshold = 0, method = "ml"),silent = TRUE))){
      #If yes lparent=TRUE
      lparent=TRUE
      fitparent=evir::gpd(y, threshold = 0, method = "ml")
    }else{
      #Else lparent=FALSE
      lparent=FALSE
    }
  }else{
    #Else lparent=FALSE
    lparent=FALSE
  }

  #Continuous x variable
  if (continuous) {
    #n is the number of observations
    n <- length(y)
    #Initialisation of the goodness and the direction vectors at 0
    #Meaning that, for now, all of the splits are useless
    goodness=double(n-1)
    direction=goodness

    for (i in 1:(n-1)){

      #Test if the evaluation of losses for the resulating children nodes have to be made
      #It is the case, if lparent=TRUE, if each children nodes gathered at least 21 observations and if the fevd splits on the children nodes work well

      if( #lparent &
        length(1:i) > 5 &
        length((i+1):n) > 5 &
        !has_warning(try(evir::gpd(y[1:i], threshold = 0, method = "ml"),silent = TRUE)) &
        !has_warning(try(evir::gpd(y[(i+1):n], threshold = 0, method = "ml"),silent = TRUE))     ){

        if(
          !has_error(try(evir::gpd(y[1:i], threshold = 0, method = "ml"),silent = TRUE)) &
          !has_error(try(evir::gpd(y[(i+1):n], threshold = 0, method = "ml"),silent = TRUE))
        ){

          #If yes, computation of the fevd fit on children nodes and attribution of the resulting goodness and direction

          fitmoins=evir::gpd(y[1:i], threshold = 0, method = "ml")
          fitplus=evir::gpd(y[(i+1):n], threshold = 0, method = "ml")

          #Reminder: the more the value of the goodness is closed to 0 the less purely the node is)
          #if(fitmoins$par.ests[1]>0 & fitplus$par.ests[1]>0){
            if(lparent==TRUE & fitmoins$par.ests[1]>0 & fitplus$par.ests[1]>0){
              goodness[i]=max(0,positive_nll(fitparent$nllh.final)-(positive_nll(fitmoins$nllh.final) + positive_nll(fitplus$nllh.final)))
            }else{
              goodness[i]=0
              #goodness[i]=max(0,naive_ll_gpd_fit(y)-(fitmoins$nllh.final + fitplus$nllh.final))
            }
          #}else{
          #  goodness[i]=0
          #}
          #goodness[i]=1/(fitmoins$nllh.final + fitplus$nllh.final) #+ 10*length(y))
          #goodness[i]=1/(fitmoins$nllh.final + fitplus$nllh.final + 10*length(y))
          direction[i]=sign(positive_nll(fitmoins$nllh.final)-positive_nll(fitplus$nllh.final))

        }
      }

      if(goodness[i]<0){
        print("Warning: negative goodness")
      }

      if(goodness[i]==Inf | goodness[i]==-Inf){
        print("Warning: + or - infinity goodness")
      }

    }

    list(goodness = goodness, direction = direction)
  } else {

    #Categorial x variable

    #We order the categories by their means
    ux <- sort(unique(x))
    ymean <- tapply(y, x, mean)
    ord <- order(ymean)
    #n is the number of categories
    n <- length(ord)

    #Initialisation of the goodness and the direction vectors at 0
    #Meaning that, for now, all of the splits are useless
    goodness=double(n-1)
    direction=goodness

    for (i in 1:(n-1)){
      #Creation of two vectors ymoins and yplus gathering respectively observations at the left (resp. at the right ) of the split
      ymoins=y[x==ux[ord[1]]]
      j=2
      while (j<=i){
        ymoins=c(ymoins,y[x==ux[ord[j]]])
        j=j+1
      }

      yplus=y[x==ux[ord[i+1]]]
      j=2
      while(i+j<=n){
        yplus=c(yplus,y[x==ux[ord[i+j]]])
        j=j+1
      }


      #Test if the evaluation of losses for the resulating children nodes have to be made
      #It is the case, if lparent=TRUE, if each children nodes gathered at least 21 observations and if the fevd splits on the children nodes work well


      if(
        #lparent &
        length(ymoins) > 5 &
        length(yplus) > 5 &
        !has_warning(try(evir::gpd(ymoins, threshold = 0, method = "ml"),silent = TRUE)) &
        !has_warning(try(evir::gpd(yplus, threshold = 0, method = "ml"),silent = TRUE))
      ){

        if(
          !has_error(try(evir::gpd(ymoins, threshold = 0, method = "ml"),silent = TRUE)) &
          !has_error(try(evir::gpd(yplus, threshold = 0, method = "ml"),silent = TRUE))     ){

          #If yes, computation of the fevd fit on children nodes and attribution of the resulting goodness and direction

          fitmoins=evir::gpd(ymoins, threshold = 0, method = "ml")
          fitplus=evir::gpd(yplus, threshold = 0, method = "ml")

          #Reminder: the more the value of the goodness is closed to 0 the less purely the node is)

          #if(fitmoins$par.ests[1]>0 & fitplus$par.ests[1]>0){
            if(lparent==TRUE & fitmoins$par.ests[1]>0 & fitplus$par.ests[1]>0){
              goodness[i]=max(0,positive_nll(evir::gpd(y, threshold = 0, method = "ml")$nllh.final)-(positive_nll(fitmoins$nllh.final) + positive_nll(fitplus$nllh.final)))
            }else{
              goodness[i]=0
              #goodness[i]=max(0,naive_ll_gpd_fit(y)-(fitmoins$nllh.final + fitplus$nllh.final))
            }
          #}else{
          #  goodness[i]=0
          #}

          #goodness[i]=evir::gpd(y, threshold = 0, method = "ml")$nllh.final-(fitmoins$nllh.final + fitplus$nllh.final)
          #goodness[i]=1/(fitmoins$nllh.final + fitplus$nllh.final) #+ 10*length(y))
          #goodness[i]=1/(fitmoins$nllh.final + fitplus$nllh.final + 10*length(y))
          direction[i]=sign(positive_nll(fitmoins$nllh.final)-positive_nll(fitplus$nllh.final))

        }
      }

      if(goodness[i]<0){
        print("Warning: negative goodness")
      }

      if(goodness[i]==Inf | goodness[i]==-Inf){
        print("Warning: + or - infinity goodness or rate of fit distinct of 1")
      }

    }

    list(goodness= goodness, direction = ux[ord])
  }
}

GPRT_method_xi_in_R_plus=list(eval = etempgprt, split = stempgprt, init = itempgprt)

#Functions for pruning

#'@param y a numeric vector
#'@param sh a shape parameter
#'@param sc a scale parameter
#'@return goodness the negative log likelihood computed on y according to a generalized pareto distribution with paramters shape, scale and location=0
LL=function(y,sh,sc){
  temp=evir::dgpd(y,xi=sh,beta=sc,mu=0)
  goodness=-(sum(log(temp)))
  if(is.na(goodness)){
    #print("Warning: unable to compute the LL for a subset of the test sample in a leaf")
    goodness=+Inf
  }
  return(goodness)
}

#'@param arbre a tree of the class rpart
#'@param train_data a test data sample
#'@return a vector of the x relative errors of the tree, ie the ratio between the sum of the negative log likelihood at each leaf and the root negative log likelihood, for each tree pruned thanks to the complexity parameter and with the test sample
construct_x_rel_error_alpha_gprt=function(arbre,test_data,name_response,alpha){
  #tree=prune(arbre,cp=alpha/arbre$frame[1,"dev"])
  tree=prune_tree(rpart_object=arbre,CP=alpha)

  y_val_tree=matrix(c(1:dim(tree$frame)[1],
                      as.numeric(row.names(tree$frame)),
                      as.numeric(as.matrix(tree$frame)[,c("yval2.1","yval2.2")])),
                    nrow=dim(tree$frame)[1],ncol=4,byrow=FALSE)
  colnames(y_val_tree)=c("where","node","sh","sc")

  number_of_terminal_nodes=sum(tree$frame[,"var"]=="<leaf>")

  where_test_data=as.vector(rpart.predict.leaves(tree, test_data, type = "where"))
  unique_where_test_data=sort(unique(where_test_data))
  unique_where_test_data=matrix(c(unique_where_test_data,rep(0,times=length(unique_where_test_data))),ncol=2,byrow=FALSE)
  colnames(unique_where_test_data)=c("where","NLL")

  for(i in 1:dim(unique_where_test_data)[1]){
    where=unique_where_test_data[i]
    y=test_data[where_test_data==where,name_response]
    # unique_where_test_data[i,2]=positive_nll(LL(y,
    #                                sh=y_val_tree[y_val_tree[,"where"]==where,"sh"],
    #                                sc=y_val_tree[y_val_tree[,"where"]==where,"sc"]))
    unique_where_test_data[i,2]=LL(y,
                                   sh=y_val_tree[y_val_tree[,"where"]==where,"sh"],
                                   sc=y_val_tree[y_val_tree[,"where"]==where,"sc"])
  }

  return(sum(unique_where_test_data[,"NLL"]))

}


#'@param arbre a tree of the class rpart
#'@param train_data a test data sample
#'@return a vector of the x relative errors of the tree, ie the ratio between the sum of the negative log likelihood at each leaf and the root negative log likelihood, for each tree pruned thanks to the complexity parameter and with the test sample
cross_val_gprt=function(data,formula,arbre,n_fold,seed,choice="first increase"){

  #On the tree

  #alpha=as.numeric(arbre$cptable[,"CP"])*arbre$frame[1,"dev"]
  alpha=as.numeric(get_table(arbre)[,"CP"])

  if(alpha[length(alpha)]==0){
    alpha=alpha[1:(length(alpha)-1)]
  }
  # if(length(alpha)>1){
  #   beta=c()
  #   for(i in 2:length(alpha)){
  #     beta=c(beta,sqrt(alpha[i]*alpha[i-1]))
  #   }
  #   beta=c(beta, alpha[length(alpha)]/2, 0)
  #   beta=c(Inf, seq(alpha[1]-0.001,beta[1],length.out=4), beta[2:length(beta)])
  # }else{
  #   beta=c(Inf, seq(alpha[1]-0.001,0,length.out=4))
  # }

  beta=sort(c(1,seq(alpha[length(alpha)],alpha[1],length.out=1000),0),decreasing=TRUE)

  #On the n_fold trees

  set.seed(seed)
  xgroup=sample(1:n_fold,size=dim(data)[1],replace=TRUE)

  # set.seed(seed)
  # xgroup=double(dim(data)[1])
  # xgroup=c(rep(sample(1:n_fold,3,replace=FALSE),length.out=3*dim(data)[1]%/%n_fold),sample(1:n_fold,dim(data)[1]%%n_fold,replace=TRUE))[order(data$X)]

    # set.seed(seed)
    # xgroup=double(dim(data)[1])
    # strat=c(0,quantile(data$X,probs=seq(0,1,length.out=dim(data)[1]%/%10))[2:(dim(data)[1]%/%10 -1)],1)
    # for(i in 1:(dim(data)[1]%/%10)){
    #   match_obs=data$X>=strat[i] & data$X<strat[i+1]
    #   if(sum(match_obs)>0){
    #     obs=data$X[match_obs]
    #     xgroup[match_obs]=sample(c(rep(1:n_fold,sum(match_obs) %/% n_fold),sample(1:n_fold,sum(match_obs) %% n_fold,replace=TRUE)),size=sum(match_obs),replace=FALSE)
    #   }
    # }


  list_train_tree=list()
  Risk_matrix=matrix(nrow = length(beta),ncol = n_fold)

  for(k in 1:n_fold){
    train_data=data[xgroup!=k,]
    test_data=data[xgroup==k,]

    tree_train=rpart(data = train_data,
                     formula = formula,
                     method = GPRT_method_xi_in_R_plus,
                     control = rpart.control(cp=0,minsplit=100, minbucket=50, maxcompete=0, maxsurrogate=0),
                     xval=0)

    list_train_tree[[k]]=tree_train

    # for(l in 1:length(beta)){
    #   Risk_matrix[l,k]=construct_x_rel_error_alpha_gprt(arbre = tree_train,test_data = test_data,name_response = as.character(formula[[2]]),alpha = beta[l])
    # }
    Risk_matrix[,k]=sapply(X = beta,FUN="construct_x_rel_error_alpha_gprt",arbre = tree_train,test_data = test_data,name_response = as.character(formula[[2]]))


  }

  Risk_beta=rowSums(Risk_matrix)
  #res=beta[Risk_beta==min(Risk_beta)]
  #We choose the deepth just before which the test error begin to increase
  pos=match(TRUE,Risk_beta[-1]>Risk_beta[-length(Risk_beta)])
  if(is.na(pos) | choice=="min"){
    #If the test error does not increase, we choose the minimal deepth that minimize the test error
    depth=beta[match(TRUE,Risk_beta==min(Risk_beta))][1]
  }else{
    depth=beta[pos][1]
  }

  res=list(depth,beta,Risk_beta)

  return(res)
}



#'@param arbre a tree of the class rpart
#'@param train_data a test data sample
#'@return a vector of the x relative errors of the tree, ie the ratio between the sum of the negative log likelihood at each leaf and the root negative log likelihood, for each tree pruned thanks to the complexity parameter and with the test sample
prune_train_test_gprt=function(data_train,data_test,formula,arbre_apprentissage){

  #On the tree

  alpha=c(as.numeric(arbre_apprentissage$cptable[,"CP"])*arbre_apprentissage$frame[1,"dev"])

  Risk_alpha=double(length(alpha))

  for(l in 1:length(alpha)){
    Risk_alpha[l]=construct_x_rel_error_alpha_gprt(arbre = arbre_apprentissage,test_data = data_test,name_response = as.character(formula[[2]]),alpha = alpha[l])
  }

  #Compute the risk for each beta and each fold


  h=Risk_alpha[1]-min(Risk_alpha,na.rm=TRUE)
  #res = max(match(TRUE, Risk_alpha <= min(Risk_alpha,na.rm=TRUE)+1/2*h)-1,1)
  res = match(TRUE, Risk_alpha == min(Risk_alpha,na.rm=TRUE))

  #res=alpha[match(TRUE,Risk_alpha==min(Risk_alpha))]

  return(res)
}


positive_nll=function(x){
  if(x>0){
    return(x+1)
  }else{
    return(exp(x))
  }
}

ll_ratio_test_prune_gprt=function(arbre){
  ll=c()
  n_param=c()
  depth=dim(arbre$cptable)[1]
  for(i in 1:depth){
    sub_tree=prune(arbre,cp=arbre$cptable[i,"CP"])
    ll=c(ll,-sum(sub_tree$frame[sub_tree$frame$var=="<leaf>","dev"]))
  }
  n_param=(arbre$cptable[,"nsplit"]+1)*2
  valeurs=ll[2:length(ll)]-ll[1:(length(ll)-1)]
  seuils=qchisq(p = 0.95,df = n_param[2:length(n_param)]-n_param[1:(length(n_param)-1)])
  decision=valeurs>seuils
  level=sum(cumsum(decision)==1:length(decision))+1
  alpha=arbre$cptable[level,"CP"]*arbre$frame[1,"dev"]
  return(alpha)
}


get_table=function(rpart_object){
  n_split=1:nrow(rpart_object$splits)
  var=rownames(rpart_object$splits)
  improve=rpart_object$splits[,"improve"]
  n=rpart_object$splits[,"count"]
  CP=(rpart_object$frame[1,"dev"]-cumsum(rpart_object$splits[,"improve"]))/rpart_object$frame[1,"dev"]
  return(cbind(n_split,CP,improve,var,n))
}

match_id_id_bis=function(id_bis,table,frame){
  frame$improve=NA
  for(i in 1:nrow(frame)){
      if(frame$var[i]!="<leaf>"){
        id=row.names(frame)[i]
        frame$improve[i]=frame$dev[row.names(frame) == as.character(as.numeric(id))]-(frame$dev[row.names(frame) == as.character(2*as.numeric(id))] + frame$dev[row.names(frame) == as.character(2*as.numeric(id) + 1)])
      }
  }

  id=rownames(frame)[frame$var==table[id_bis,"var"] & frame$n==table[id_bis,"n"]  &  round(as.numeric(frame$improve),digits=6)==round(as.numeric(table[id_bis,"improve"]),digits=6)]
  return(as.numeric(id))
}

prune_tree=function(rpart_object,CP){
  frame <- rpart_object$frame

  COMPUTE=FALSE
  if(length(rpart_object$splits)>0){
    if(nrow(rpart_object$splits)>0){
      COMPUTE=TRUE
    }
  }
  if(COMPUTE){

    table=get_table(rpart_object)
    id_bis_splits_to_remove=table[,"n_split"][table[,"CP"]<CP]

    if(length(id_bis_splits_to_remove)==nrow(table)){
      return(prune(rpart_object,cp=Inf))
    }else{
      if(length(id_bis_splits_to_remove)==0){
        return(rpart_object)
      }else{
        id_splits_to_remove=sapply(as.numeric(id_bis_splits_to_remove),FUN="match_id_id_bis",table=table,frame=frame)
        return(snip.rpart(rpart_object, id_splits_to_remove))
      }
    }

  }else{
    return(rpart_object)
  }

}

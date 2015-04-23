library(textir)
library(data.table)

## read in the text
revs <- read.table("data/yelp_phrases.txt",
	sep="|",quote=NULL, comment="", 
	col.names=c("id","phrase","stars","sample"))

x <- sparseMatrix( 
			i=revs[,"id"]+1, j=as.numeric(revs[,"phrase"]), x=rep(1,nrow(revs)),
			dimnames=list(NULL, levels(revs[,"phrase"])) )
emptyrev <- which(rowSums(x)==0)
x <- x[-emptyrev,colSums(x>0)>5]

stars <- tapply(revs$stars, revs$id, mean)
samp <- tapply( revs$sample=="test", revs$id, mean)

### COARSE logit word-count prediction
coarse <- which( stars %in% c(1,2,5) )
xcoarse <- x[coarse, ]
xcoarse <- xcoarse[, colSums(xcoarse>0)>5]
ycoarse <- as.numeric(stars[coarse]>2)
testcoarse <- which(samp[coarse]==1)

coarsefit <- function(x, y, testset, ...){
    fit = gamlr(x[-testset,], y[-testset], ...)
    phat <- predict(fit, x[testset,], type="response")
    yhat <- as.numeric(phat > .5)
    cat("\nMCR bad: ", mean(yhat[y[testset]==0]), 
	   ", good: ", 1-mean(yhat[y[testset]==1]), 
	   ", overall: ", mean(y[testset]!=yhat))
    dev <- sum(-2*log(phat[y[testset]==1])) + sum(-2*log(1-phat[y[testset]==0]))
    cat("  DEV: ", dev/length(testset), "\n")
    return(fit)
}

logitcoarse <- coarsefit(xcoarse, ycoarse, testcoarse, 
    verb=TRUE, family="binomial", lambda.min.ratio=1e-4)
# MCR bad:  0.07883443 , good:  0.02898715 , overall:  0.04567453  DEV:  0.2674133

png(file="graphs/yelp_logistic.png", width=12,height=5, units="in", res=360)
plot(logitcoarse)
invisible(dev.off())

nosd <- coarsefit(xcoarse, ycoarse, testcoarse, 
    verb=TRUE, family="binomial", lambda.min.ratio=1e-2, standardize=FALSE)
# MCR bad:  0.2990739 , good:  0.05877004 , overall:  0.1392166  DEV:  0.6760135

## D2V stuff
dv0train <- fread("data/yelpD2Vtrain0.csv")
dv0test <- fread("data/yelpD2Vtest0.csv")
dv1train <- fread("data/yelpD2Vtrain1.csv")
dv1test <- fread("data/yelpD2Vtest1.csv")
all(dv0test[,id]==dv1test[,id])
all(dv0test[,stars]==dv1test[,stars])
vecvar <- paste("x",1:100,sep="")
dv0x <- rbind(as.matrix(dv0train[,vecvar,with=FALSE]),
            as.matrix(dv0test[,vecvar,with=FALSE]))
dv1x <- rbind(as.matrix(dv1train[,vecvar,with=FALSE]),
            as.matrix(dv1test[,vecvar,with=FALSE]))
dvx <- cbind(dv0x,dv1x)
dvstars <- c(dv0train[,stars], dv0test[,stars])
dvsamp <- c(rep(FALSE, nrow(dv0train)),rep(TRUE, nrow(dv0test)))

# create coarse dv designs
dvcoarse <- which(dvstars %in% c(1,2,5))
dv0xcoarse <- dv0x[dvcoarse, ]
dv1xcoarse <- dv1x[dvcoarse, ]
dvxcoarse <- dvx[dvcoarse, ]
dvycoarse <- as.numeric(dvstars[dvcoarse]>2)
dvtestcoarse <- which(dvsamp[dvcoarse])

## all run at zero lambda; AICc selects most complex model anyways
dv0fit <- coarsefit(dv0xcoarse, dvycoarse, dvtestcoarse,
    verb=TRUE, family="binomial", lambda.start=0)
# MCR bad:  0.2118103 , good:  0.02004377 , overall:  0.08441111  DEV:  0.4359189 
dv1fit <- coarsefit(dv1xcoarse, dvycoarse, dvtestcoarse,
    verb=TRUE, family="binomial", lambda.start=0)
# MCR bad:  0.5102599 , good:  0.01866144 , overall:  0.1836688  DEV:  0.7584851 
dvfit <- coarsefit(dvxcoarse, dvycoarse, dvtestcoarse,
    verb=TRUE, family="binomial", lambda.start=0)
# MCR bad:  0.2026904 , good:  0.02061974 , overall:  0.08173261  DEV:  0.4349279 

### mnir
library(parallel)
cl <- makeCluster(4, type="FORK")

fitir <- mnlm(cl=cl, ycoarse[-testcoarse], xcoarse[-testcoarse,], verb=1, bins=2)
z <- srproj(fitir, xcoarse, select=100)
fwd <- gamlr(z[-testcoarse,], ycoarse[-testcoarse], family="binomial", lambda.start=0)
phatir <- predict(fwd, z[testcoarse,], type="response")
yhatir <- as.numeric(  phatir > 0.5 )
dev <- sum(-2*log(phatir[ycoarse[testcoarse]==1])) + 
            sum(-2*log(1-phatir[ycoarse[testcoarse]==0]))
cat("\nMCR bad: ", mean(yhatir[ycoarse[testcoarse]==0]), 
       ", good: ", 1-mean(yhatir[ycoarse[testcoarse]==1]), 
       ", overall: ", mean(ycoarse[testcoarse]!=yhatir),
        "  DEV: ", dev/length(testcoarse), "\n")
# MCR bad:  0.08018974 , good:  0.05479141 , overall:  0.06329401   DEV:  0.3439997

### FINE logit word-count prediction

testset <- which(samp==1)
y <- factor(stars)

finefit <- function(x, y, testset, ...){
    fit = dmr(cl=cl, x[-testset,], y[-testset], ...)
    phat <- predict(fit, x[testset,], type="response")
    yhat <- apply(phat,1,which.max)

    cat("\nMCR ")
    for(k in  1:5)
        cat(k, ":", mean(yhat[y[testset]==k]!=k), ", ")
    cat("\noverall: ", mean(y[testset]!=yhat), ".  ")

    pofy <- phat[cbind(1:length(testset),y[testset])]
    dev <- sum(-2*log(pofy))/length(testset)
    cat("DEVIANCE =",dev, "\n")
 

    return(fit)
}

logitfine <- finefit(x, y, testset, lambda.min.ratio=1e-4)
logitcoarse <- finefit(csx, y, testset,  lambda.min.ratio=1e-4)
# MCR 1 : 0.4802521 ,2 : 0.753786 ,3 : 0.7738764 ,4 : 0.3591457 ,5 : 0.2329203 , 
# overall:  0.409976 .  DEVIANCE = 2.023658
finenosd <- finefit(x, y, testset, lambda.min.ratio=1e-4, standardize=FALSE)
# MCR 1 : 0.4327731 ,2 : 0.7132389 ,3 : 0.7426264 ,4 : 0.3594363 ,5 : 0.2318972 , 
# overall:  0.3972555 .  DEVIANCE = 1.9518 

dvtest <- which(dvsamp)
dvy <- factor(dvstars)

## all run at zero lambda; AICc selects most complex model anyways
dv0fit <- finefit(dv0x, dvy, dvtest, lambda.min.ratio=1e-4)
# MCR 1 : 0.7687075 ,2 : 0.9587021 ,3 : 0.9680738 ,4 : 0.1788869 ,5 : 0.4119341 
# overall:  0.4973102 .  DEVIANCE = 2.312074
dv1fit <- finefit(dv1x, dvy, dvtest, lambda.min.ratio=1e-4)
# MCR 1 : 0.9277211 ,2 : 0.9936087 ,3 : 0.9950337 ,4 : 0.1890459 ,5 : 0.4924548 , 
# overall:  0.5541494 .  DEVIANCE = 2.557209 
dvfit <- finefit(dvx, dvy, dvtest, lambda.min.ratio=1e-4)
# MCR 1 : 0.8001701 ,2 : 0.9646018 ,3 : 0.9702022 ,4 : 0.1682862 ,5 : 0.4316323 
# overall:  0.5057324 .  DEVIANCE = 2.327374 

ymat <- sparse.model.matrix(~stars + y-1)
fitirfine <- mnlm(cl=cl, ymat[-testset,], x[-testset,], verb=1, bins=2)
zfine <- srproj(fitirfine, x, select=100)
fwdfine <- dmr(cl, zfine[-testset,], y[-testset], lambda.min.ratio=1e-4)
pirfine <- predict(fwdfine, zfine[testset,], type="response")
yirfine <- apply(pirfine,1,which.max)

cat("\nMCR ")
for(k in  1:5)
   print(mean(yirfine[y[testset]==k]!=k))
cat("\noverall: ", mean(y[testset]!=yirfine), ".  ")
pofyfine <- pirfine[cbind(1:length(testset),y[testset])]
devfine <- sum(-2*log(pofyfine))/length(testset)
cat("DEVIANCE =",devfine, "\n")
# MCR 
# [1] 0.6105042
# [1] 0.9706888
# [1] 0.9827949
# [1] 0.2949295
# [1] 0.3124929
# overall:  0.4799826. DEVIANCE = 2.298992 






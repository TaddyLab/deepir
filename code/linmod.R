suppressMessages(library(textir))
suppressMessages(library(data.table))

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
test <- which(samp==1)

## read d2v
dv0train <- fread("data/yelpD2Vtrain0.csv", verbose=FALSE)
dv0test <- fread("data/yelpD2Vtest0.csv", verbose=FALSE)
dv1train <- fread("data/yelpD2Vtrain1.csv", verbose=FALSE)
dv1test <- fread("data/yelpD2Vtest1.csv", verbose=FALSE)
# all(dv0test[,id]==dv1test[,id])
# all(dv0test[,stars]==dv1test[,stars])
vecvar <- paste("x",1:100,sep="")
dv0x <- rbind(as.matrix(dv0train[,vecvar,with=FALSE]),
            as.matrix(dv0test[,vecvar,with=FALSE]))
dv1x <- rbind(as.matrix(dv1train[,vecvar,with=FALSE]),
            as.matrix(dv1test[,vecvar,with=FALSE]))
dvx <- cbind(dv0x,dv1x)
dvstars <- c(dv0train[,stars], dv0test[,stars])
dvtest <- nrow(dv0train)+1:nrow(dv0test)

library(parallel)
cl <- makeCluster(6, type="FORK")

geterr <- function(phat, y, PY=FALSE){
    if(ncol(phat)==1) phat <- cbind(1-phat,phat)
    y <- factor(y)
    yhat <- factor(levels(y)[apply(phat,1,which.max)])
    cat("mcr ")
    for(l in levels(y))
        cat(l, ":", round(
            mean(yhat[y==l] != y[y==l]),3), ", ", sep="")
    overall <- mean(yhat !=y)
    diff <- mean( abs(as.numeric(yhat) - as.numeric(y)) )
    py <- phat[cbind(1:nrow(phat),y)]
    lp <- log(py)
    lp[lp < (-50)] <- -50
    dev <- mean(-2*lp)
    cat("\noverall:", round(overall,3), "diff:", round(diff,3), "deviance:", dev, "\n")
    if(PY) return(py)
    invisible()
} 

getpy <- function(fit, xx, y, testset, PY=FALSE){
    if(inherits(fit,"randomForest"))
        phat <- as.matrix(predict(fit, xx[testset,], type="prob"))
    else 
        phat <- predict(fit, xx[testset,], type="response")
    py <- geterr(phat, y[testset], PY=PY)
    if(PY) return(py) 
    invisible()
}

## define y
ycoarse <- as.numeric(stars>2)
ynnp <- cut(stars, c(0,2,3,5))
yfine <- factor(stars)
dvycoarse <- as.numeric(dvstars==5)
dvynnp <- cut(dvstars, c(0,2,3,5))
dvyfine <- factor(dvstars)

### W2V inversion
cat("\n**** W2V INVERSION ****\n")
w2vprob <- fread("data/fineyelpscore.csv", header=TRUE, verbose=FALSE)
w2vprob <- as.matrix(w2vprob[-emptyrev,])

nullprob <- as.numeric(table(stars[-test])/length(stars[-test]))
n <- nrow(w2vprob)

cat("** COARSE **\n")
w2vpcoarse <- cbind(rowSums(w2vprob[,1:2]),rowSums(w2vprob[,3:5]))
geterr(w2vpcoarse[test,], ycoarse[test])

cat("** NNP **\n")
w2vpnnp <- cbind(rowSums(w2vprob[,1:2]),
    rowSums(w2vprob[,3,drop=FALSE]),
    rowSums(w2vprob[,4:5,drop=FALSE]))
geterr(w2vpnnp[test,], ynnp[test])

cat("** FINE **\n")
geterr(w2vprob[test,], yfine[test])

### COARSE logit word-count prediction
cat("\n*** COUNTREG ***\n")

cat("** COARSE **\n")
logitcoarse <- gamlr(x[-test,], ycoarse[-test], 
                family="binomial")
pycoarse <- getpy(logitcoarse, x, ycoarse, test, PY=TRUE)

png(file="graphs/yelp_logistic.png", width=12,height=5, units="in", res=360)
plot(logitcoarse)
invisible(dev.off())

cat("** NNP **\n")
logitnnp <- dmr(cl=cl, x[-test,], ynnp[-test])
pynnp <- getpy(logitnnp, x, ynnp, test, PY=TRUE)

cat("** FINE **\n")
logitfine <- dmr(cl=cl, x[-test,], yfine[-test])
pyfine <- getpy(logitfine, x, yfine, test, PY=TRUE)

cat("\n*** W2V and COUNTREG NNP ***\n")
wx <- cBind(w2vprob,x)
combof <- dmr(cl,wx[-test,], ynnp[-test])
getpy(combof, wx, ynnp, test)

## D2V stuff
## all run at zero lambda; AICc selects most complex model anyways
cat("\n*** D2V ***\n")

cat("** COARSE\n")
cat("dm0 **\n")
dv0coarse <- gamlr(dv0x[-dvtest,], dvycoarse[-dvtest],
                family="binomial", lambda.start=0)
getpy(dv0coarse, dv0x, dvycoarse, dvtest)
cat("dm1 **\n")
dv1coarse <- gamlr(dv1x[-dvtest,], dvycoarse[-dvtest],
                family="binomial", lambda.start=0)
getpy(dv1coarse, dv1x, dvycoarse, dvtest)
cat("dm both **\n")
dvcoarse <- gamlr(dvx[-dvtest,], dvycoarse[-dvtest],
                family="binomial", lambda.start=0)
getpy(dvcoarse, dvx, dvycoarse, dvtest)

cat("** NNP\n")
cat("dm0 **\n")
dv0nnp <- gamlr(dv0x[-dvtest,], dvynnp[-dvtest],
                family="binomial", lambda.start=0)
getpy(dv0nnp, dv0x, dvynnp, dvtest)
cat("dm1 **\n")
dv1nnp <- gamlr(dv1x[-dvtest,], dvynnp[-dvtest],
                family="binomial", lambda.start=0)
getpy(dv1nnp, dv1x, dvynnp, dvtest)
cat("dm both **\n")
dvnnp <- gamlr(dvx[-dvtest,], dvynnp[-dvtest],
                family="binomial", lambda.start=0)
getpy(dvnnp, dvx, dvynnp, dvtest)

cat("** FINE\n")
cat("dm0 **\n")
dv0fine <- gamlr(dv0x[-dvtest,], dvyfine[-dvtest],
                family="binomial", lambda.start=0)
getpy(dv0fine, dv0x, dvyfine, dvtest)
cat("dm1 **\n")
dv1fine <- gamlr(dv1x[-dvtest,], dvyfine[-dvtest],
                family="binomial", lambda.start=0)
getpy(dv1fine, dv1x, dvyfine, dvtest)
cat("dm both **\n")
dvfine <- gamlr(dvx[-dvtest,], dvyfine[-dvtest],
                family="binomial", lambda.start=0)
getpy(dvfine, dvx, dvyfine, dvtest)

# mnir
cat("\n*** MNIR ***\n")
vmat <- sparse.model.matrix(~stars + yfine-1)
mnir <- mnlm(cl=cl, vmat[-test,], x[-test,], verb=1, bins=5)
zir <- srproj(mnir, x, select=100)

cat("** COARSE **\n")
fwdcoarse <- dmr(cl, zir[-test,], ycoarse[-test], lambda.min.ratio=1e-4)
getpy(fwdcoarse, zf, ycoarse, test)

cat("** NNP **\n")
fwdnnp <- dmr(cl, zir[-test,], ynnp[-test], lambda.min.ratio=1e-4)
getpy(fwdnnp, zf, ynnp, test)

cat("** FINE **\n")
fwdfine <- dmr(cl, zir[-test,], yfine[-test], lambda.min.ratio=1e-4)
getpy(fwdfine, zf, yfine, test)


### some plots
w2vpc <- w2vpcoarse[test,2]
pyc <- pycoarse

pdf("graphs/coarseprob.pdf", width=10, height=5)
par(mfrow=c(1,2))
hist(w2vpc[ycoarse[test]==0], col=rgb(1,0,0,.7), breaks=10, freq=FALSE,
         xlab="prob(positive)", xlim=c(0,1), ylim=c(0,8), main="w2v inversion")
hist(w2vpc[ycoarse[test]==1], col=rgb(1,1,0,.7), breaks=10, freq=FALSE, add=TRUE)

hist(pyc[ycoarse[test]==0], col=rgb(1,0,0,.7), breaks=10, freq=FALSE,
         xlab="prob(positive)", xlim=c(0,1), ylim=c(0,8), main="phrase regression")
hist(pyc[ycoarse[test]==1], col=rgb(1,1,0,.7), breaks=10, freq=FALSE, add=TRUE)
dev.off()


pdf("graphs/coarseprob_bystar.pdf", width=10, height=5)
par(mfrow=c(1,2))
par(mfrow=c(1,2), mai=c(.7,.7,.5,.2), omi=c(.3,.3,0,0))
boxplot( w2vpc ~ yfine[test], col=heat.colors(5), varwidth=TRUE, main="w2v inversion")
boxplot( pyc~ yfine[test], col=heat.colors(5), varwidth=TRUE, main="phrase regression")
mtext(side=1, "stars", outer=TRUE)
mtext(side=2, "p(stars > 2)", outer=TRUE)
dev.off()

w2vpnnpy <- w2vpnnp[cbind(1:n,stars)]
pdf("graphs/nnpprob.pdf", width=8, height=4)
par(mfrow=c(1,2), mai=c(.7,.7,.5,.2), omi=c(.3,.3,0,0))
boxplot( w2vpnnpy[test] ~ ynnp[test], col=heat.colors(5), varwidth=TRUE, main="w2v inversion")
boxplot( pynnp~ ynnp[test], col=heat.colors(5), varwidth=TRUE, main="phrase regression")
mtext(side=2, "p(y)", outer=TRUE)
dev.off()

w2vpy <- w2vprob[cbind(1:n,stars)]
pdf("graphs/fineprob.pdf", width=8, height=4)
par(mfrow=c(1,2), mai=c(.7,.7,.5,.2), omi=c(.3,.3,0,0))
boxplot( w2vpy[test] ~ yfine[test], col=heat.colors(5), varwidth=TRUE, main="w2v inversion")
points(1:5, nullprob, pch=18,cex=2, col="navy")
boxplot( pyfine~ yfine[test], col=heat.colors(5), varwidth=TRUE, main="phrase regression")
points(1:5, nullprob, pch=18,cex=2, col="navy")
mtext(side=1, "stars", outer=TRUE)
mtext(side=2, "p(y)", outer=TRUE)
dev.off()

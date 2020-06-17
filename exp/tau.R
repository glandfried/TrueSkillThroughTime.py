
1/(0.3^2)

pi <- seq(0.015,20,by=0.015)
plot(pi,dgamma(pi,1,1),type="l")
plot(pi,dnorm(3,1,pi),type="l")

nc = sum(dgamma(pi,1,1)*dnorm(3,1,sqrt(1/pi) ))*0.015
plot(pi,dgamma(pi,1,1)*dnorm(3,1,sqrt(1/pi) )*(1/nc),type="l")
nc = sum(dgamma(pi,1,1)*dnorm(3,1,1+sqrt(1/pi) ))*0.015
lines(pi,dgamma(pi,1,1)*dnorm(3,1,1+sqrt(1/pi) )*(1/nc),type="l")
esperanza = sum(pi * dgamma(pi,1,1)*dnorm(3,1,1+sqrt(1/pi))*(1/nc)*0.015 )
varianza = sum(((pi-esperanza)^2) * dgamma(pi,1,1)*dnorm(3,1,1+sqrt(1/pi)*(1/nc)*0.015 ))
lines(pi,dgamma(pi,(esperanza^2)/varianza,esperanza/varianza),type="l")

v=1
nc = sum(dgamma(pi,1,1)*dnorm(3,1,v+sqrt(1/pi) ))*0.015
plot(pi, dgamma(pi,1,1)*dnorm(3,1,v+sqrt(1/pi))*(1/nc),type="l")
esperanza = sum(pi * dgamma(pi,1,1)*dnorm(3,1,v+sqrt(1/pi))*(1/nc)*0.015 )
lines(pi, dgamma(pi,1,esperanza/varianza^2))


lines(pi,dgamma(pi,3/2,3),type="l")

nc = sum(dgamma(pi,1,1)*dnorm(3,1,1+sqrt(1/pi) ))*0.015
plot(pi,dgamma(pi,1,1)*dnorm(3,1,1+sqrt(1/pi) )*(1/nc),type="l")
lines(pi,dgamma(pi,3/2,1.55),type="l")

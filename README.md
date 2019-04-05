# Airbnb Big Data Project

Estimating Best Airbnb Price Based on
Similar Postings

The purpose of this project is to estimate the optimal airbnb price for a new listing. There
will several factors to consider as different postings offer different accommodations and
amenities. Similar postings will be compared to the new posting using Apache’s PySpark for
compiling data. Afterwards, SciKit Learn Neural networks and Random Forests will be used as
our prediction generation algorithms. Finally, the training set will be dynamically chosen at
runtime based on the listing features that are fed to the system. We will be comparing the
different machine learning algorithms to each other as well as trying to optimise the parameters
of the algorithms to yield the best price estimator model.


The environment we are using is a linux docker container. 

docker run -it -v "$(pwd):/assignment" bigdatalabteam/sparkdocker
pip install sklearn

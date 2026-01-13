---
sdk: docker
app_port: 7860
---


# planparser
Architectural plan elements detection


[Lind for data downloading](https://universe.roboflow.com/research-g8szb/floorplan-details-fork/dataset/1)


docker run -d --name planparser -p 7860:7860 --env-file .env planparser
docker stop planparser
docker start planparser

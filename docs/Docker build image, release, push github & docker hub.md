# Docker build image, release, push github & docker hub

[](https://github.com/eclipse/kuksa.val.feeders/tree/main/csv_provider)

- git clone this page
- Change .csv file and change the Docker.file , [provider.py](http://provider.py)’s path too (if you modified the csv file name)
- Go to folder and command

```jsx
docker build -t [image_name] .
```

Now you can chech docker image in your local

```jsx
docker images
```

Docker images are stored in `/var/lib/docker`

(but when you access that folder , you need root authority)

Now you can release your docker image in docker hub (you should generate docker token first)

```jsx
docker login

`# Enter your docker account user named & docker token`
```

```jsx
docker push [imagename]:tag
```

If you encounter the `Error saving credentials` then go to the `~/.docker`  and delete `config.json` file

When you want to download test 0.1 version

```jsx
docker pull seungwoo1123/provider:0.1
```

1. **Generate and Use a Personal Access Token (PAT)**:
    - Create a PAT on GitHub with permissions including **`write:packages`**, **`read:packages`**, and **`delete:packages`**.
    - Use this token when logging in to the Docker CLI with your GitHub account.
2. **Tagging the Docker Image**:
    - Assign a tag for the image relevant to your organization, using the format **`ghcr.io/[organization-name]/[image-name]:[tag]`**.
    - For instance, replace **`[organization-name]`** with the name of your Organization.
    - Remember to write all names in lowercase, as case sensitivity does not matter.
    - Example command: `docker tag seungwoo1123/provider ghcr.io/[organization-name]/provider:latest`
    
    ```bash
    docker tag seungwoo1123/provider ghcr.io/[organization-name]/provider:latest
    
    ```
    
3. **Pushing to GHCR**:
    - Push the tagged image to GHCR:
    
    ```bash
    docker push ghcr.io/[organization-name]/provider:latest
    
    ```
    
    - Authenticate using the PAT created earlier during this process.
4. **Using the Image in Leda**:
    - The image uploaded to GHCR can now be pulled into the Leda project using the command **`docker pull ghcr.io/[organization-name]/provider:latest`**.

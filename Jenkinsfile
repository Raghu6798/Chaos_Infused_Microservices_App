pipeline {
    agent {
        // --- THIS IS THE KEY CHANGE ---
        // This is a powerful agent image that includes Git, Docker, and kubectl all in one.
        docker {
            image 'carlosrfr/docker-kubectl:latest'
            // We still need to give this container access to the host's Docker engine.
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }

    environment {
        DOCKERHUB_USERNAME = 'raghumaverick'
        IMAGE_NAME = "${DOCKERHUB_USERNAME}/chaos-infused-app"
        K8S_NAMESPACE = 'dev'
    }

    stages {
        // Note: The explicit 'Checkout' stage is now handled automatically by the agent,
        // but we can leave it here for clarity.
        stage('Checkout from Git') {
            steps {
                echo 'Checking out code from Git repository...'
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image: ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                sh "docker build -f app/Dockerfile -t ${IMAGE_NAME}:${env.BUILD_NUMBER} ."
                sh "docker tag ${IMAGE_NAME}:${env.BUILD_NUMBER} ${IMAGE_NAME}:latest"
            }
        }

        stage('Push to Docker Hub') {
            steps {
                echo "Logging in and pushing image to Docker Hub..."
                withCredentials([usernamePassword(credentialsId: 'DOCKERHUB_CREDENTIALS', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh "echo ${DOCKER_PASS} | docker login -u ${DOCKER_USER} --password-stdin"
                    sh "docker push ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    sh "docker push ${IMAGE_NAME}:latest"
                }
            }
        }

        // --- DEPLOYMENT STAGE IS NOW ENABLED ---
        stage('Deploy to Kubernetes') {
            steps {
                echo 'Deploying to Minikube cluster...'
                withCredentials([file(credentialsId: 'MINIKUBE_KUBECONFIG', variable: 'KUBECONFIG_FILE')]) {
                    sh """
                        export KUBECONFIG=\$KUBECONFIG_FILE
                        
                        echo 'Updating image in Kubernetes deployment...'
                        kubectl set image deployment/fastapi-app-deployment fastapi-app=${IMAGE_NAME}:${env.BUILD_NUMBER} -n ${K8S_NAMESPACE}

                        echo 'Applying service configuration...'
                        kubectl apply -f k8s/service.yaml -n ${K8S_NAMESPACE}

                        echo 'Verifying rollout status...'
                        kubectl rollout status deployment/fastapi-app-deployment -n ${K8S_NAMESPACE}
                    """
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished. Cleaning up...'
            sh 'docker logout'
        }
    }
}

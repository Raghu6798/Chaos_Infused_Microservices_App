// Jenkinsfile for Raghu's Chaos_Infused_Microservices_App

pipeline {
    // Run on the main Jenkins node. This is fine for this setup.
    agent any

    environment {
        // --- CONFIGURATION ---
        // Your Docker Hub username is set here.
        DOCKERHUB_USERNAME = 'raghumaverick'
        // The image will be named raghumaverick/chaos-infused-app
        IMAGE_NAME = "${DOCKERHUB_USERNAME}/chaos-infused-app"
        // The Kubernetes namespace for deployment.
        K8S_NAMESPACE = 'dev'
    }

    stages {
        stage('Checkout from Git') {
            steps {
                echo 'Checking out code from Git repository...'
                // This step pulls the code from the job's Git configuration.
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image: ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                
                // --- CUSTOMIZED BUILD COMMAND ---
                // This command correctly points to your Dockerfile inside the 'app' directory
                // -f app/Dockerfile : Specifies the path to the Dockerfile.
                // .                 : Sets the build context to the project root.
                sh "docker build -f app/Dockerfile -t ${IMAGE_NAME}:${env.BUILD_NUMBER} ."
                
                // Tag the new build as 'latest' for convenience
                sh "docker tag ${IMAGE_NAME}:${env.BUILD_NUMBER} ${IMAGE_NAME}:latest"
            }
        }

        stage('Push to Docker Hub') {
            steps {
                echo "Logging in and pushing image to Docker Hub..."
                // Use the credential ID 'DOCKERHUB_CREDENTIALS' you configured in Jenkins
                withCredentials([usernamePassword(credentialsId: 'DOCKERHUB_CREDENTIALS', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh "echo ${DOCKER_PASS} | docker login -u ${DOCKER_USER} --password-stdin"
                    // Push the uniquely tagged version
                    sh "docker push ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    // Push the 'latest' tag
                    sh "docker push ${IMAGE_NAME}:latest"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo 'Deploying to Minikube cluster...'
                // Use the kubeconfig credential ID 'MINIKUBE_KUBECONFIG' you configured in Jenkins
                withCredentials([file(credentialsId: 'MINIKUBE_KUBECONFIG', variable: 'KUBECONFIG_FILE')]) {
                    sh """
                        export KUBECONFIG=\$KUBECONFIG_FILE
                        
                        echo 'Updating image in Kubernetes deployment...'
                        # This performs a safe, zero-downtime rolling update of your application
                        # It tells the deployment to use the new image we just built and pushed.
                        kubectl set image deployment/fastapi-app-deployment fastapi-app=${IMAGE_NAME}:${env.BUILD_NUMBER} -n ${K8S_NAMESPACE}

                        echo 'Applying service configuration to ensure it is up-to-date...'
                        kubectl apply -f k8s/service.yaml -n ${K8S_NAMESPACE}

                        echo 'Verifying rollout status...'
                        kubectl rollout status deployment/fastapi-app-deployment -n ${K8S_NAMESPACE}
                    """
                }
            }
        }
    }

    post {
        // This block runs after all stages, regardless of success or failure.
        always {
            echo 'Pipeline finished. Cleaning up...'
            // Good practice to log out of Docker.
            sh 'docker logout'
        }
    }
}

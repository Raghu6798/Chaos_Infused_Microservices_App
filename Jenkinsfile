pipeline {
    agent any // We will run directly on the Jenkins controller node

    environment {
        // --- CONFIGURATION ---
        // Change these to match your setup
        DOCKERHUB_USERNAME = 'raghumaverick'
        IMAGE_NAME = "${DOCKERHUB_USERNAME}/fastapi-chaos-app"
        K8S_NAMESPACE = 'dev'
    }

    stages {
        stage('Prepare Workspace') {
            steps {
                echo 'Cleaning up old workspace...'
                // Clean the workspace to ensure we're not using old files
                cleanWs()
                
                echo 'Copying application files into workspace...'
                // This assumes your project is located at C:\Users\Raghu\Downloads\Incidence_response_agent
                // !!! IMPORTANT: CHANGE THIS PATH TO MATCH YOUR PROJECT LOCATION !!!
                dir('C:Users/Raghu/Downloads/Incidence_response_agent/Grafana_Loki_test') {
                    // We copy the necessary files into the Jenkins job's workspace
                    sh 'cp -r app k8s Dockerfile .'
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo "Building Docker image: ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                // Use the build number as a unique tag
                sh "docker build -t ${IMAGE_NAME}:${env.BUILD_NUMBER} ."
                sh "docker tag ${IMAGE_NAME}:${env.BUILD_NUMBER} ${IMAGE_NAME}:latest"
            }
        }

        stage('Push to Docker Hub') {
            steps {
                echo "Logging in and pushing image to Docker Hub..."
                // Use the credential ID you configured in Jenkins
                withCredentials([usernamePassword(credentialsId: 'DOCKERHUB_CREDENTIALS', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh "echo ${DOCKER_PASS} | docker login -u ${DOCKER_USER} --password-stdin"
                    sh "docker push ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    sh "docker push ${IMAGE_NAME}:latest"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo 'Deploying to Minikube cluster...'
                // Use the kubeconfig credential ID you configured in Jenkins
                withCredentials([file(credentialsId: 'MINIKUBE_KUBECONFIG', variable: 'KUBECONFIG_FILE')]) {
                    sh """
                        export KUBECONFIG=\$KUBECONFIG_FILE
                        
                        echo 'Updating image in Kubernetes deployment...'
                        # This command updates the deployment to use the new image version
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
using System;
using UnityEngine;
using UnityEngine.UIElements;

struct Particle {
    public Vector2 position;
    public Vector2 velocity;

    public float mass;
    public float density;
    public float pressure;
}

public class SPHController : MonoBehaviour
{
    public bool simRunning = true;
    public bool renderTexture = true;
    public ComputeShader simulationShader;
    public ComputeShader renderShader;
    private ComputeBuffer[] particleBuffers = new ComputeBuffer[2];
    private int currentBufferId = 0;

    public int particleCount = 100;
    public int latticeWidth = 10;
    private Particle[] particleArray;
    public Material renderMaterial;

    private int updateKernelIndex;
    private int advectionKernelIndex;
    private int renderKernelIndex;
    
    public float pressureStiffness = 5;
    public float restDensity = 1;
    public float kernelRadius = 2;
    public float viscosityCoefficient = 1;

    public float renderSmoothingLength = 5;

    public float domainWidth = 50;
    public float domainHeight = 50;

    private RenderTexture outputTexture;

    // Shader Dispatch

    void updateBuffers() {
        int nextBufferId = (currentBufferId + 1) % 2;

        simulationShader.SetBuffer(updateKernelIndex, "particles", particleBuffers[currentBufferId]);
        simulationShader.SetBuffer(advectionKernelIndex, "particles", particleBuffers[currentBufferId]);

        simulationShader.SetBuffer(updateKernelIndex, "particlesNew", particleBuffers[nextBufferId]);
        simulationShader.SetBuffer(advectionKernelIndex, "particlesNew", particleBuffers[nextBufferId]);
    }

    void swapBuffers() {
        currentBufferId = (currentBufferId + 1) % 2;
        updateBuffers();
    }

    Vector2 worldToDomainPoint(Vector3 worldPos) {
        float sizeScale = Mathf.Max(domainHeight,domainWidth);
        return (Vector2)transform.InverseTransformPoint(worldPos)*sizeScale + Vector2.up*sizeScale*0.5f;
    }

    void stepSimulation(float dt) {
        int numGroups = Mathf.CeilToInt((float)particleCount / 128);

        // User Interaction

        Vector2 mousePos = Vector2.zero;
        if (Input.GetMouseButton(0)) {
            Vector3 mouseWorldPosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mousePos = worldToDomainPoint(mouseWorldPosition);
        }

        // Setting uniforms
        simulationShader.SetFloat("pressureStiffness",pressureStiffness);
        simulationShader.SetFloat("restDensity",restDensity);
        simulationShader.SetFloat("h",kernelRadius);
        simulationShader.SetFloat("nu",viscosityCoefficient);

        simulationShader.SetFloat("domainWidth",domainWidth);
        simulationShader.SetFloat("domainHeight",domainHeight);

        simulationShader.SetFloat("DeltaTime",dt);
        simulationShader.SetVector("mousePos",mousePos);
        simulationShader.SetInt("particleCount",particleCount);

        // Running Compute Shader
        simulationShader.SetFloat("DeltaTime",dt/2);
        simulationShader.Dispatch(advectionKernelIndex, numGroups, 1, 1);
        swapBuffers();

        simulationShader.SetFloat("DeltaTime",dt);
        simulationShader.Dispatch(updateKernelIndex, numGroups, 1, 1);
        swapBuffers();

        simulationShader.SetFloat("DeltaTime",dt/2);
        simulationShader.Dispatch(advectionKernelIndex, numGroups, 1, 1);
        swapBuffers();
    }

    // Rendering
    bool IsMatrixValid(Matrix4x4 matrix) {
        for (int i = 0; i < 16; i++) {
            if (!float.IsFinite(matrix[i])) {
                return false;
            }
        }
        return true;
    }

    void RenderToTexture() {
        int numGroups = Mathf.CeilToInt((float)particleCount / 128);

        renderShader.SetFloat("domainWidth",domainWidth);
        renderShader.SetFloat("domainHeight",domainHeight);
        renderShader.SetFloat("h",renderSmoothingLength);
        renderShader.SetBuffer(renderKernelIndex,"particles", particleBuffers[currentBufferId]);
        renderShader.SetTexture(renderKernelIndex, "textureOut", outputTexture);

        renderShader.Dispatch(renderKernelIndex, outputTexture.width / 8, outputTexture.height / 8, 1);

        renderMaterial.mainTexture = outputTexture;
    }

    // Stepping

    void Update() {
        if (Input.GetKeyDown(KeyCode.Space)) {
            simRunning = !simRunning;
        }

        if (Input.GetKeyDown(KeyCode.Alpha1)) {
            renderShader.SetInt("renderMode",1);
        } else if (Input.GetKeyDown(KeyCode.Alpha2)) {
            renderShader.SetInt("renderMode",2);
        } else if (Input.GetKeyDown(KeyCode.Alpha3)) {
            renderShader.SetInt("renderMode",3);
        }

        if (Input.GetKeyDown(KeyCode.S)) {
            string fileName = "Screenshot.png";
            ScreenCapture.CaptureScreenshot(fileName, 1);

            string rootPath = Application.dataPath.Substring(0, Application.dataPath.Length - "/Assets".Length);
            string filePath = rootPath + "/" + fileName;
            Debug.Log("Captured screenshot. Storing at " + filePath);
        }

        if (simRunning) {
            stepSimulation(Time.deltaTime);
        }
        
        RenderToTexture();
    }

    // Initialisation and Cleanup

    void Start()
    {
        particleArray = new Particle[particleCount];

        for (int i = 0; i < particleCount; i++) {
            Particle p = new Particle();

            p.mass = 1;
            p.density = 1;
            p.pressure = 0;

            int x = i % latticeWidth - latticeWidth/2;
            int y = i / latticeWidth;
            p.position = new Vector2(x,y);
            p.velocity = Vector2.zero;

            particleArray[i] = p;
        }
        
        for (int i = 0; i < 2; i++) {
            particleBuffers[i] = new ComputeBuffer(particleCount, sizeof(float) * 7);
            particleBuffers[i].SetData(particleArray);
        }

        updateKernelIndex = simulationShader.FindKernel("updateParticles");
        advectionKernelIndex = simulationShader.FindKernel("advectParticles");

        renderKernelIndex = renderShader.FindKernel("renderParticles");
        renderShader.SetInt("renderMode",1);

        outputTexture = new RenderTexture(1024,1024,1);
        outputTexture.enableRandomWrite = true;
        outputTexture.Create();

        updateBuffers(); // Ensure buffers are set properly
    }

    void OnDestroy() {
        for (int i = 0; i < 2; i++) {
            particleBuffers[i].Release();
        }
    }
}

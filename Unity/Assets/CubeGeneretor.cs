using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeGeneretor : MonoBehaviour
{
    public GameObject cubePrefab;
    float spawnInterval = 0.6f; //doğma süresi
    float nextSpawnTime = 0f;

    void Start()
    {
        nextSpawnTime = Time.time + spawnInterval;
    }

    void Update()
    {
        if (Time.time >= nextSpawnTime)
        {
            Instantiate(cubePrefab, Vector3.one, Quaternion.identity);
            nextSpawnTime = Time.time + spawnInterval;
        }
    }
}

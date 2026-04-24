<!-- frontend/src/App.vue -->
<template>
  <div id="app">
    <el-container class="app-container">
      <!-- Sidebar -->
      <el-aside width="200px" class="sidebar">
        <div class="logo">
          <el-icon><Flask /></el-icon>
          <span>ChemAI Nexus</span>
        </div>
        <el-menu :default-active="activeMenu" router mode="vertical">
          <el-menu-item index="/data">
            <el-icon><Folder /></el-icon>
            <span>Data</span>
          </el-menu-item>
          <el-menu-item index="/pipeline">
            <el-icon><Setting /></el-icon>
            <span>Pipeline</span>
          </el-menu-item>
          <el-menu-item index="/results">
            <el-icon><Document /></el-icon>
            <span>Results</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <!-- Main Content -->
      <el-container>
        <el-header class="app-header">
          <h2 class="text-xl font-bold">ChemAI Nexus v2.0</h2>
          <div class="header-tags">
            <el-tag v-if="store.hasData" type="success" size="small">
              Data Loaded: {{ store.filename }}
            </el-tag>
            <el-tag v-else type="info" size="small">No Data</el-tag>
          </div>
        </el-header>
        <el-main class="app-main">
          <router-view :key="$route.fullPath" />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Flask, Folder, Setting, Document } from '@element-plus/icons-vue'
import { useChemaiStore } from './store/chemai'

const route = useRoute()
const store = useChemaiStore()
const activeMenu = computed(() => route.path)
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, #app { height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.app-container { height: 100%; }
.sidebar { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); color: white; border-right: none; }
.logo { display: flex; align-items: center; gap: 10px; padding: 20px; font-size: 18px; font-weight: bold; border-bottom: 1px solid rgba(255,255,255,0.1); }
.el-menu { background: transparent; border-right: none; }
.el-menu-item { color: rgba(255,255,255,0.8) !important; }
.el-menu-item.is-active { background: rgba(255,255,255,0.1) !important; color: #42b983 !important; }
.el-menu-item:hover { background: rgba(255,255,255,0.05) !important; }
.app-header { background: #f5f7fa; border-bottom: 1px solid #e4e7ed; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; height: 64px; }
.app-main { background: #f0f2f5; padding: 20px; overflow-y: auto; }
.header-tags { display: flex; gap: 10px; }
</style>

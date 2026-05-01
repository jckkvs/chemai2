<!-- frontend/src/App.vue -->
<template>
  <el-container class="app-container">
    <!-- Sidebar Sidebar -->
    <el-aside width="260px" class="sidebar">
      <div class="logo-section">
        <h1 class="logo-text">ChemAI<span>Nexus</span></h1>
      </div>
      
      <el-menu
        :default-active="activeRoute"
        class="side-menu"
        background-color="transparent"
        text-color="#c0c4cc"
        active-text-color="#ffffff"
        router
      >
        <el-menu-item index="/data">
          <el-icon><DataLine /></el-icon>
          <span>Data Management</span>
        </el-menu-item>
        
        <el-menu-item index="/eda">
          <el-icon><PieChart /></el-icon>
          <span>Exploratory Analysis</span>
        </el-menu-item>
        
        <el-menu-item index="/ml">
          <el-icon><Cpu /></el-icon>
          <span>ML Studio</span>
        </el-menu-item>
        
        <el-menu-item index="/results">
          <el-icon><TrendCharts /></el-icon>
          <span>Results Report</span>
        </el-menu-item>
      </el-menu>

      <div class="sidebar-footer">
        <div class="version">v2.0.0 Stable</div>
      </div>
    </el-aside>

    <!-- Main Content -->
    <el-container class="main-container">
      <el-header class="app-header">
        <div class="header-left">
          <el-breadcrumb separator="/">
            <el-breadcrumb-item>Workspace</el-breadcrumb-item>
            <el-breadcrumb-item>{{ currentViewName }}</el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        <div class="header-right">
          <el-button link icon="QuestionFilled">Documentation</el-button>
          <el-avatar size="small" icon="UserFilled" />
        </div>
      </el-header>

      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="fade-transform" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const activeRoute = computed(() => route.path)

const currentViewName = computed(() => {
  const map: Record<string, string> = {
    '/data': 'Data Management',
    '/eda': 'Exploratory Analysis',
    '/ml': 'ML Studio',
    '/results': 'Results Report'
  }
  return map[route.path] || 'Home'
})
</script>

<style>
:root {
  --sidebar-bg: #1a1c1e;
  --header-bg: #ffffff;
  --main-bg: #f5f7fa;
  --primary-color: #409eff;
}

body {
  margin: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.app-container {
  height: 100vh;
  overflow: hidden;
}

/* Sidebar Styling */
.sidebar {
  background-color: var(--sidebar-bg);
  display: flex;
  flex-direction: column;
  border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.logo-section {
  padding: 30px 24px;
}

.logo-text {
  color: white;
  margin: 0;
  font-size: 1.5rem;
  letter-spacing: -0.5px;
  font-weight: 800;
}

.logo-text span {
  color: var(--primary-color);
  font-weight: 300;
}

.side-menu {
  border-right: none;
  flex: 1;
}

.side-menu .el-menu-item {
  height: 56px;
  margin: 4px 12px;
  border-radius: 8px;
}

.side-menu .el-menu-item.is-active {
  background-color: var(--primary-color) !important;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3);
}

.sidebar-footer {
  padding: 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  color: #606266;
  font-size: 0.8rem;
}

/* Header Styling */
.app-header {
  background-color: var(--header-bg);
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #ebeef5;
  padding: 0 30px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

/* Main Content Area */
.main-container {
  background-color: var(--main-bg);
}

.app-main {
  padding: 30px;
  overflow-y: auto;
}

/* Transitions */
.fade-transform-enter-active,
.fade-transform-leave-active {
  transition: all 0.3s;
}

.fade-transform-enter-from {
  opacity: 0;
  transform: translateX(-10px);
}

.fade-transform-leave-to {
  opacity: 0;
  transform: translateX(10px);
}
</style>
